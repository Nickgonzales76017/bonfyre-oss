/*
 * BonfyreAuth — user management and session tokens.
 *
 * Manages signup, login, sessions, and ties users to Gate API keys.
 *
 * Schema:
 *   users: id, email, name, password_hash, salt, tier, created_at, active
 *   sessions: id, user_id, token, created_at, expires_at, active
 *
 * Password hashing: SHA-256(salt + password) — no external deps.
 *
 * Usage:
 *   bonfyre-auth signup --email E --name N --password P [--tier free|pro|enterprise]
 *   bonfyre-auth login --email E --password P
 *   bonfyre-auth verify --token T
 *   bonfyre-auth logout --token T
 *   bonfyre-auth users
 *   bonfyre-auth user --id ID
 *   bonfyre-auth update --id ID [--tier T] [--active 0|1]
 *   bonfyre-auth sessions --user-id ID
 *   bonfyre-auth status
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <sqlite3.h>
#include <bonfyre.h>

#define MAX_PATH  2048
#define MAX_TOKEN 128
#define SESSION_HOURS 168  /* 7 days */

static const char g_hex_lut[16] = "0123456789abcdef";

static void hash_password(const char *salt, const char *password, char *out64) {
    char combined[512];
    snprintf(combined,sizeof(combined),"%s:%s",salt,password);
    bf_sha256_hex((const uint8_t*)combined,strlen(combined),out64);
}

static void generate_salt(char *salt, size_t sz) {
    FILE *f=fopen("/dev/urandom","rb");
    uint8_t bytes[16];
    if (f) { fread(bytes,1,sizeof(bytes),f); fclose(f); }
    else { for(int i=0;i<16;i++) bytes[i]=(uint8_t)(rand()&0xff); }
    int n=16<(int)(sz/2)?16:(int)(sz/2);
    for(int i=0;i<n;i++){
        salt[i*2]  =g_hex_lut[bytes[i]>>4];
        salt[i*2+1]=g_hex_lut[bytes[i]&0x0f];
    }
    salt[n*2]='\0';
}

static void generate_token(char *token, size_t sz) {
    FILE *f=fopen("/dev/urandom","rb");
    uint8_t bytes[32];
    if (f) { fread(bytes,1,sizeof(bytes),f); fclose(f); }
    else { for(int i=0;i<32;i++) bytes[i]=(uint8_t)(rand()&0xff); }
    /* bfy_ prefix for easy identification */
    int off=4;
    if(sz<5){token[0]='\0';return;}
    memcpy(token,"bfy_",4);
    for(int i=0;i<32&&off+2<(int)sz;i++){
        token[off++]=g_hex_lut[bytes[i]>>4];
        token[off++]=g_hex_lut[bytes[i]&0x0f];
    }
    token[off]='\0';
}

/* ── Utility ──────────────────────────────────────────────────────── */

static void iso_now(char *buf, size_t sz) {
    time_t t=time(NULL); struct tm tm; gmtime_r(&t,&tm);
    strftime(buf,sz,"%Y-%m-%dT%H:%M:%SZ",&tm);
}

static void iso_future(char *buf, size_t sz, int hours) {
    time_t t=time(NULL)+hours*3600;
    struct tm tm; gmtime_r(&t,&tm);
    strftime(buf,sz,"%Y-%m-%dT%H:%M:%SZ",&tm);
}

/* ── Database ─────────────────────────────────────────────────────── */

static const char *SCHEMA_SQL =
    "CREATE TABLE IF NOT EXISTS users ("
    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  email TEXT UNIQUE NOT NULL,"
    "  name TEXT NOT NULL,"
    "  password_hash TEXT NOT NULL,"
    "  salt TEXT NOT NULL,"
    "  tier TEXT NOT NULL DEFAULT 'free',"
    "  gate_key TEXT,"
    "  created_at TEXT NOT NULL,"
    "  active INTEGER NOT NULL DEFAULT 1"
    ");"
    "CREATE TABLE IF NOT EXISTS sessions ("
    "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
    "  user_id INTEGER NOT NULL,"
    "  token TEXT UNIQUE NOT NULL,"
    "  created_at TEXT NOT NULL,"
    "  expires_at TEXT NOT NULL,"
    "  active INTEGER NOT NULL DEFAULT 1,"
    "  FOREIGN KEY (user_id) REFERENCES users(id)"
    ");";

static sqlite3 *open_db(const char *path) {
    sqlite3 *db;
    if (bf_sqlite3_open(path,&db)!=SQLITE_OK) {
        fprintf(stderr,"Cannot open %s: %s\n",path,sqlite3_errmsg(db)); return NULL;
    }
    char *err=NULL;
    if (sqlite3_exec(db,SCHEMA_SQL,NULL,NULL,&err)!=SQLITE_OK) {
        fprintf(stderr,"Schema error: %s\n",err);
        sqlite3_free(err); sqlite3_close(db); return NULL;
    }
    return db;
}

static const char *default_db(void) {
    static char p[MAX_PATH];
    const char *h=getenv("HOME");
    snprintf(p,sizeof(p),"%s/.local/share/bonfyre/auth.db",h?h:".");
    char d[MAX_PATH];
    snprintf(d,sizeof(d),"%s/.local/share/bonfyre",h?h:".");
    mkdir(d,0755);
    return p;
}

/* ── Commands ─────────────────────────────────────────────────────── */

static int cmd_signup(sqlite3 *db, const char *email, const char *name,
                      const char *password, const char *tier) {
    /* Check duplicate */
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,"SELECT id FROM users WHERE email=?",-1,&st,NULL);
    sqlite3_bind_text(st,1,email,-1,SQLITE_STATIC);
    if (sqlite3_step(st)==SQLITE_ROW) {
        fprintf(stderr,"Email already registered: %s\n",email);
        sqlite3_finalize(st); return 1;
    }
    sqlite3_finalize(st);

    char salt[64]; generate_salt(salt,sizeof(salt));
    char phash[128]; hash_password(salt,password,phash);
    char ts[64]; iso_now(ts,sizeof(ts));

    /* Issue a Gate key for this user */
    char gate_key[MAX_TOKEN]; generate_token(gate_key,sizeof(gate_key));

    sqlite3_prepare_v2(db,
        "INSERT INTO users (email,name,password_hash,salt,tier,gate_key,created_at) VALUES (?,?,?,?,?,?,?)",
        -1,&st,NULL);
    sqlite3_bind_text(st,1,email,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,2,name,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,3,phash,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,4,salt,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,5,tier,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,6,gate_key,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,7,ts,-1,SQLITE_STATIC);
    if (sqlite3_step(st)!=SQLITE_DONE) {
        fprintf(stderr,"Signup failed: %s\n",sqlite3_errmsg(db));
        sqlite3_finalize(st); return 1;
    }
    sqlite3_finalize(st);

    int uid=(int)sqlite3_last_insert_rowid(db);
    printf("{\"user\":{\"id\":%d,\"email\":\"%s\",\"name\":\"%s\",\"tier\":\"%s\"},",
        uid,email,name,tier);
    printf("\"gate_key\":\"%s\",\"created\":true}\n",gate_key);
    return 0;
}

static int cmd_login(sqlite3 *db, const char *email, const char *password) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT id,name,password_hash,salt,tier,gate_key,active FROM users WHERE email=?",
        -1,&st,NULL);
    sqlite3_bind_text(st,1,email,-1,SQLITE_STATIC);
    if (sqlite3_step(st)!=SQLITE_ROW) {
        fprintf(stderr,"User not found: %s\n",email);
        sqlite3_finalize(st); return 1;
    }

    int uid=sqlite3_column_int(st,0);
    const char *name_v=(const char*)sqlite3_column_text(st,1);
    const char *stored_hash=(const char*)sqlite3_column_text(st,2);
    const char *salt=(const char*)sqlite3_column_text(st,3);
    const char *tier_v=(const char*)sqlite3_column_text(st,4);
    const char *gate_v=(const char*)sqlite3_column_text(st,5);
    int active=sqlite3_column_int(st,6);

    if (!active) {
        fprintf(stderr,"Account disabled.\n");
        sqlite3_finalize(st); return 1;
    }

    char check[128]; hash_password(salt,password,check);
    if (strcmp(check,stored_hash)!=0) {
        fprintf(stderr,"Invalid password.\n");
        sqlite3_finalize(st); return 1;
    }

    /* Copy values before finalize */
    char name_buf[256], tier_buf[64], gate_buf[MAX_TOKEN];
    snprintf(name_buf,sizeof(name_buf),"%s",name_v?name_v:"");
    snprintf(tier_buf,sizeof(tier_buf),"%s",tier_v?tier_v:"free");
    snprintf(gate_buf,sizeof(gate_buf),"%s",gate_v?gate_v:"");
    sqlite3_finalize(st);

    /* Create session */
    char token[MAX_TOKEN]; generate_token(token,sizeof(token));
    char ts[64]; iso_now(ts,sizeof(ts));
    char exp[64]; iso_future(exp,sizeof(exp),SESSION_HOURS);

    sqlite3_prepare_v2(db,
        "INSERT INTO sessions (user_id,token,created_at,expires_at) VALUES (?,?,?,?)",
        -1,&st,NULL);
    sqlite3_bind_int(st,1,uid);
    sqlite3_bind_text(st,2,token,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,3,ts,-1,SQLITE_STATIC);
    sqlite3_bind_text(st,4,exp,-1,SQLITE_STATIC);
    sqlite3_step(st); sqlite3_finalize(st);

    printf("{\"session\":{\"token\":\"%s\",\"expires_at\":\"%s\"},",token,exp);
    printf("\"user\":{\"id\":%d,\"email\":\"%s\",\"name\":\"%s\",\"tier\":\"%s\",\"gate_key\":\"%s\"}}\n",
        uid,email,name_buf,tier_buf,gate_buf);
    return 0;
}

static int cmd_verify(sqlite3 *db, const char *token) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT s.user_id,s.expires_at,s.active,u.email,u.name,u.tier,u.gate_key,u.active "
        "FROM sessions s JOIN users u ON s.user_id=u.id WHERE s.token=?",
        -1,&st,NULL);
    sqlite3_bind_text(st,1,token,-1,SQLITE_STATIC);
    if (sqlite3_step(st)!=SQLITE_ROW) {
        printf("{\"valid\":false,\"error\":\"session not found\"}\n");
        sqlite3_finalize(st); return 1;
    }

    int uid=sqlite3_column_int(st,0);
    const char *exp=(const char*)sqlite3_column_text(st,1);
    int sess_active=sqlite3_column_int(st,2);
    const char *email_v=(const char*)sqlite3_column_text(st,3);
    const char *name_v=(const char*)sqlite3_column_text(st,4);
    const char *tier_v=(const char*)sqlite3_column_text(st,5);
    const char *gate_v=(const char*)sqlite3_column_text(st,6);
    int user_active=sqlite3_column_int(st,7);

    if (!sess_active || !user_active) {
        printf("{\"valid\":false,\"error\":\"session or account disabled\"}\n");
        sqlite3_finalize(st); return 1;
    }

    /* Check expiry */
    char now[64]; iso_now(now,sizeof(now));
    if (strcmp(now,exp?exp:"")>0) {
        printf("{\"valid\":false,\"error\":\"session expired\"}\n");
        sqlite3_finalize(st); return 1;
    }

    printf("{\"valid\":true,\"user\":{\"id\":%d,\"email\":\"%s\",\"name\":\"%s\","
           "\"tier\":\"%s\",\"gate_key\":\"%s\"}}\n",
        uid,email_v?email_v:"",name_v?name_v:"",tier_v?tier_v:"free",gate_v?gate_v:"");
    sqlite3_finalize(st);
    return 0;
}

static int cmd_logout(sqlite3 *db, const char *token) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,"UPDATE sessions SET active=0 WHERE token=?",-1,&st,NULL);
    sqlite3_bind_text(st,1,token,-1,SQLITE_STATIC);
    sqlite3_step(st); sqlite3_finalize(st);
    printf("{\"logged_out\":true}\n");
    return 0;
}

static int cmd_users(sqlite3 *db) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT id,email,name,tier,created_at,active FROM users ORDER BY id",-1,&st,NULL);
    printf("%-4s %-30s %-20s %-10s %-8s %s\n","ID","Email","Name","Tier","Active","Created");
    printf("------------------------------------------------------------------------------------\n");
    while (sqlite3_step(st)==SQLITE_ROW) {
        printf("%-4d %-30s %-20s %-10s %-8s %s\n",
            sqlite3_column_int(st,0),
            (const char*)sqlite3_column_text(st,1),
            (const char*)sqlite3_column_text(st,2),
            (const char*)sqlite3_column_text(st,3),
            sqlite3_column_int(st,5)?"yes":"no",
            (const char*)sqlite3_column_text(st,4));
    }
    sqlite3_finalize(st);
    return 0;
}

static int cmd_user(sqlite3 *db, int uid) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT id,email,name,tier,gate_key,created_at,active FROM users WHERE id=?",
        -1,&st,NULL);
    sqlite3_bind_int(st,1,uid);
    if (sqlite3_step(st)!=SQLITE_ROW) {
        fprintf(stderr,"User #%d not found.\n",uid);
        sqlite3_finalize(st); return 1;
    }
    printf("{\"user\":{\"id\":%d,\"email\":\"%s\",\"name\":\"%s\","
           "\"tier\":\"%s\",\"gate_key\":\"%s\",\"created_at\":\"%s\",\"active\":%s}}\n",
        sqlite3_column_int(st,0),
        (const char*)sqlite3_column_text(st,1),
        (const char*)sqlite3_column_text(st,2),
        (const char*)sqlite3_column_text(st,3),
        sqlite3_column_text(st,4)?(const char*)sqlite3_column_text(st,4):"",
        (const char*)sqlite3_column_text(st,5),
        sqlite3_column_int(st,6)?"true":"false");
    sqlite3_finalize(st);
    return 0;
}

static int cmd_update(sqlite3 *db, int uid, const char *tier, const char *active_s) {
    if (tier) {
        sqlite3_stmt *st;
        sqlite3_prepare_v2(db,"UPDATE users SET tier=? WHERE id=?",-1,&st,NULL);
        sqlite3_bind_text(st,1,tier,-1,SQLITE_STATIC);
        sqlite3_bind_int(st,2,uid);
        sqlite3_step(st); sqlite3_finalize(st);
    }
    if (active_s) {
        int active=atoi(active_s);
        sqlite3_stmt *st;
        sqlite3_prepare_v2(db,"UPDATE users SET active=? WHERE id=?",-1,&st,NULL);
        sqlite3_bind_int(st,1,active);
        sqlite3_bind_int(st,2,uid);
        sqlite3_step(st); sqlite3_finalize(st);
    }
    printf("User #%d updated.\n",uid);
    return cmd_user(db,uid);
}

static int cmd_sessions(sqlite3 *db, int uid) {
    sqlite3_stmt *st;
    sqlite3_prepare_v2(db,
        "SELECT id,token,created_at,expires_at,active FROM sessions WHERE user_id=? ORDER BY id DESC",
        -1,&st,NULL);
    sqlite3_bind_int(st,1,uid);
    printf("%-4s %-20s %-22s %-22s %s\n","ID","Token (prefix)","Created","Expires","Active");
    printf("------------------------------------------------------------------------\n");
    while (sqlite3_step(st)==SQLITE_ROW) {
        const char *tok=(const char*)sqlite3_column_text(st,1);
        printf("%-4d %-20.*s... %-22s %-22s %s\n",
            sqlite3_column_int(st,0),
            16,tok,
            (const char*)sqlite3_column_text(st,2),
            (const char*)sqlite3_column_text(st,3),
            sqlite3_column_int(st,4)?"yes":"no");
    }
    sqlite3_finalize(st);
    return 0;
}

static int cmd_status(sqlite3 *db) {
    int total_users=0, active_users=0, total_sessions=0, active_sessions=0;
    int free_t=0, pro_t=0, ent_t=0;
    sqlite3_stmt *st;

    sqlite3_prepare_v2(db,"SELECT COUNT(*),SUM(active) FROM users",-1,&st,NULL);
    if (sqlite3_step(st)==SQLITE_ROW) {
        total_users=sqlite3_column_int(st,0);
        active_users=sqlite3_column_int(st,1);
    }
    sqlite3_finalize(st);

    sqlite3_prepare_v2(db,"SELECT COUNT(*),SUM(active) FROM sessions",-1,&st,NULL);
    if (sqlite3_step(st)==SQLITE_ROW) {
        total_sessions=sqlite3_column_int(st,0);
        active_sessions=sqlite3_column_int(st,1);
    }
    sqlite3_finalize(st);

    sqlite3_prepare_v2(db,"SELECT tier,COUNT(*) FROM users GROUP BY tier",-1,&st,NULL);
    while (sqlite3_step(st)==SQLITE_ROW) {
        const char *t=(const char*)sqlite3_column_text(st,0);
        int c=sqlite3_column_int(st,1);
        if (strcmp(t,"free")==0) free_t=c;
        else if (strcmp(t,"pro")==0) pro_t=c;
        else if (strcmp(t,"enterprise")==0) ent_t=c;
    }
    sqlite3_finalize(st);

    printf("BonfyreAuth Status\n");
    printf("  Users:       %d total, %d active\n",total_users,active_users);
    printf("  Tiers:       free=%d, pro=%d, enterprise=%d\n",free_t,pro_t,ent_t);
    printf("  Sessions:    %d total, %d active\n",total_sessions,active_sessions);
    return 0;
}

/* ── CLI ──────────────────────────────────────────────────────────── */

static const char *arg_get(int argc, char **argv, const char *flag) {
    for (int i=0;i<argc-1;i++)
        if (strcmp(argv[i],flag)==0) return argv[i+1];
    return NULL;
}

static void usage(void) {
    fprintf(stderr,
        "BonfyreAuth — user management & sessions\n\n"
        "Usage:\n"
        "  bonfyre-auth [--db FILE] signup --email E --name N --password P [--tier free|pro|enterprise]\n"
        "  bonfyre-auth [--db FILE] login --email E --password P\n"
        "  bonfyre-auth [--db FILE] verify --token T\n"
        "  bonfyre-auth [--db FILE] logout --token T\n"
        "  bonfyre-auth [--db FILE] users\n"
        "  bonfyre-auth [--db FILE] user --id ID\n"
        "  bonfyre-auth [--db FILE] update --id ID [--tier T] [--active 0|1]\n"
        "  bonfyre-auth [--db FILE] sessions --user-id ID\n"
        "  bonfyre-auth [--db FILE] status\n"
        "  bonfyre-auth layer-source <artifact_id> [--root DIR]\n");
}

int main(int argc, char **argv) {
    if (argc<2) { usage(); return 1; }

    const char *db_path=arg_get(argc,argv,"--db");
    if (!db_path) db_path=default_db();

    /* Strip --db and its arg */
    int ca=0; char *cv[128];
    for (int i=0;i<argc&&ca<128;i++) {
        if (strcmp(argv[i],"--db")==0){i++;continue;}
        cv[ca++]=argv[i];
    }
    if (ca<2) { usage(); return 1; }

    const char *cmd=cv[1];

    if (strcmp(cmd, "layer-source") == 0 && ca >= 3) {
        const char *root = arg_get(ca, cv, "--root");
        char *json = NULL;
        if (bf_layer_load_json(root, cv[2], &json) != 0) {
            fprintf(stderr, "unknown artifact: %s\n", cv[2]);
            return 1;
        }
        {
            char *out = NULL;
            int rc = bf_layer_auth_source_json(json, &out);
            free(json);
            if (rc != 0 || !out) return 1;
            puts(out);
            free(out);
            return 0;
        }
    }

    sqlite3 *db=open_db(db_path);
    if (!db) return 1;
    int rc=0;

    if (strcmp(cmd,"signup")==0) {
        const char *email=arg_get(ca,cv,"--email");
        const char *name=arg_get(ca,cv,"--name");
        const char *pw=arg_get(ca,cv,"--password");
        const char *tier=arg_get(ca,cv,"--tier");
        if (!tier) tier="free";
        if (!email||!name||!pw) { fprintf(stderr,"Missing --email, --name, --password\n"); rc=1; }
        else rc=cmd_signup(db,email,name,pw,tier);
    } else if (strcmp(cmd,"login")==0) {
        const char *email=arg_get(ca,cv,"--email");
        const char *pw=arg_get(ca,cv,"--password");
        if (!email||!pw) { fprintf(stderr,"Missing --email, --password\n"); rc=1; }
        else rc=cmd_login(db,email,pw);
    } else if (strcmp(cmd,"verify")==0) {
        const char *tok=arg_get(ca,cv,"--token");
        if (!tok) { fprintf(stderr,"Missing --token\n"); rc=1; }
        else rc=cmd_verify(db,tok);
    } else if (strcmp(cmd,"logout")==0) {
        const char *tok=arg_get(ca,cv,"--token");
        if (!tok) { fprintf(stderr,"Missing --token\n"); rc=1; }
        else rc=cmd_logout(db,tok);
    } else if (strcmp(cmd,"users")==0) {
        rc=cmd_users(db);
    } else if (strcmp(cmd,"user")==0) {
        const char *id_s=arg_get(ca,cv,"--id");
        if (!id_s) { fprintf(stderr,"Missing --id\n"); rc=1; }
        else rc=cmd_user(db,atoi(id_s));
    } else if (strcmp(cmd,"update")==0) {
        const char *id_s=arg_get(ca,cv,"--id");
        const char *tier=arg_get(ca,cv,"--tier");
        const char *active=arg_get(ca,cv,"--active");
        if (!id_s) { fprintf(stderr,"Missing --id\n"); rc=1; }
        else rc=cmd_update(db,atoi(id_s),tier,active);
    } else if (strcmp(cmd,"sessions")==0) {
        const char *uid_s=arg_get(ca,cv,"--user-id");
        if (!uid_s) { fprintf(stderr,"Missing --user-id\n"); rc=1; }
        else rc=cmd_sessions(db,atoi(uid_s));
    } else if (strcmp(cmd,"status")==0) {
        rc=cmd_status(db);
    } else {
        fprintf(stderr,"Unknown command: %s\n",cmd); usage(); rc=1;
    }

    sqlite3_close(db);
    return rc;
}
