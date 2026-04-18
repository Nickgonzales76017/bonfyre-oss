/*
 * BonfyreRecipe — pipeline recipe registry.
 *
 * Content-addressed registry of pipeline recipes (stage DAGs).
 * Recipes are JSON documents stored in SQLite at
 * ~/.local/share/bonfyre/recipes.db  (override: $BONFYRE_RECIPE_DB)
 *
 * Commands:
 *   bonfyre-recipe list                  — list all recipes (built-in + custom)
 *   bonfyre-recipe show <code>           — print full recipe JSON
 *   bonfyre-recipe add <file.json>       — register a custom recipe from file
 *   bonfyre-recipe add --builtin         — seed registry with all built-in recipes
 *   bonfyre-recipe search <query>        — full-text search over names + descriptions
 *   bonfyre-recipe validate <code>       — check all stage binaries exist on PATH
 *   bonfyre-recipe hash <code>           — print SHA-256 of recipe JSON
 *   bonfyre-recipe rm <code>             — remove a recipe
 *   bonfyre-recipe init [dir]            — write starter recipe.json to dir (default: .)
 *   bonfyre-recipe status                — registry stats
 *   bonfyre-recipe help                  — this message
 */

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <sqlite3.h>

#define VERSION       "1.0.0"
#define MAX_JSON      65536
#define HASH_HEX      65
#define DB_ENV        "BONFYRE_RECIPE_DB"
#define DB_SUBPATH    "/.local/share/bonfyre/recipes.db"

/* ====================================================================
 * SHA-256  (FIPS 180-4, no external deps)
 * ==================================================================== */

static const uint32_t K256[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
    0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
    0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
    0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
    0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
    0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
    0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
    0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
    0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define RR32(x,n)  (((x)>>(n))|((x)<<(32-(n))))
#define S0(x)  (RR32(x,2)^RR32(x,13)^RR32(x,22))
#define S1(x)  (RR32(x,6)^RR32(x,11)^RR32(x,25))
#define s0(x)  (RR32(x,7)^RR32(x,18)^((x)>>3))
#define s1(x)  (RR32(x,17)^RR32(x,19)^((x)>>10))
#define CH(e,f,g)   (((e)&(f))^((~(e))&(g)))
#define MAJ(a,b,c)  (((a)&(b))^((a)&(c))^((b)&(c)))

typedef struct { uint32_t h[8]; uint8_t buf[64]; uint64_t total; } SHA256_CTX;

static void sha256_init(SHA256_CTX *c){
    static const uint32_t H0[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    memcpy(c->h,H0,32); c->total=0;
}
static void sha256_block(SHA256_CTX *c, const uint8_t *d){
    uint32_t w[64],st[8],t1,t2;
    for(int i=0;i<16;i++) w[i]=((uint32_t)d[i*4]<<24)|((uint32_t)d[i*4+1]<<16)|
                                ((uint32_t)d[i*4+2]<<8)|d[i*4+3];
    for(int i=16;i<64;i++) w[i]=s1(w[i-2])+w[i-7]+s0(w[i-15])+w[i-16];
    memcpy(st,c->h,32);
    for(int i=0;i<64;i++){
        t1=st[7]+S1(st[4])+CH(st[4],st[5],st[6])+K256[i]+w[i];
        t2=S0(st[0])+MAJ(st[0],st[1],st[2]);
        st[7]=st[6];st[6]=st[5];st[5]=st[4];st[4]=st[3]+t1;
        st[3]=st[2];st[2]=st[1];st[1]=st[0];st[0]=t1+t2;
    }
    for(int i=0;i<8;i++) c->h[i]+=st[i];
}
static void sha256_update(SHA256_CTX *c, const uint8_t *data, size_t len){
    size_t off=(size_t)(c->total%64); c->total+=len;
    if(off>0){ size_t fill=64-off; if(len<fill){memcpy(c->buf+off,data,len);return;}
        memcpy(c->buf+off,data,fill); sha256_block(c,c->buf); data+=fill; len-=fill; }
    while(len>=64){sha256_block(c,data);data+=64;len-=64;}
    if(len>0) memcpy(c->buf,data,len);
}
static void sha256_final(SHA256_CTX *c, uint8_t out[32]){
    uint64_t bits=c->total*8; size_t off=(size_t)(c->total%64);
    c->buf[off++]=0x80;
    if(off>56){while(off<64)c->buf[off++]=0;sha256_block(c,c->buf);off=0;}
    while(off<56)c->buf[off++]=0;
    for(int i=7;i>=0;i--) c->buf[56+(7-i)]=(uint8_t)(bits>>(i*8));
    sha256_block(c,c->buf);
    for(int i=0;i<8;i++){out[i*4]=(uint8_t)(c->h[i]>>24);out[i*4+1]=(uint8_t)(c->h[i]>>16);
                         out[i*4+2]=(uint8_t)(c->h[i]>>8);out[i*4+3]=(uint8_t)(c->h[i]);}
}
static void sha256_hex(const char *text, size_t len, char hex[HASH_HEX]){
    SHA256_CTX ctx; sha256_init(&ctx);
    sha256_update(&ctx,(const uint8_t*)text,len);
    uint8_t h[32]; sha256_final(&ctx,h);
    static const char LUT[]="0123456789abcdef";
    for(int i=0;i<32;i++){hex[i*2]=LUT[h[i]>>4];hex[i*2+1]=LUT[h[i]&0xf];}
    hex[64]='\0';
}

/* ====================================================================
 * Minimal JSON helpers — targeted at recipe schema only
 * ==================================================================== */

/* Skip whitespace */
static const char *js_ws(const char *p){ while(*p==' '||*p=='\t'||*p=='\n'||*p=='\r')p++; return p; }

/* Extract first string value for "key" at any depth (returns 1 on success) */
static int js_str(const char *json, const char *key, char *out, size_t sz){
    char needle[128]; snprintf(needle,sizeof(needle),"\"%s\"",key);
    const char *p=json;
    while((p=strstr(p,needle))!=NULL){
        const char *v=p+strlen(needle);
        v=js_ws(v); if(*v!=':'){p++;continue;}
        v=js_ws(v+1); if(*v!='"'){p++;continue;}
        v++; size_t i=0;
        while(*v&&*v!='"'&&i<sz-1){if(*v=='\\'&&*(v+1)){v++;}out[i++]=*v++;}
        out[i]='\0'; return 1;
    }
    out[0]='\0'; return 0;
}

/* Find start of array for key, returns pointer to '[' or NULL */
static const char *js_arr_start(const char *json, const char *key){
    char needle[128]; snprintf(needle,sizeof(needle),"\"%s\"",key);
    const char *p=strstr(json,needle); if(!p) return NULL;
    p+=strlen(needle); p=js_ws(p); if(*p!=':'||!(p=js_ws(p+1))||*p!='[') return NULL;
    return p;
}

/* Skip balanced {}, [], "" value; returns pointer PAST the value */
static const char *js_skip(const char *p){
    p=js_ws(p);
    if(*p=='"'){p++; while(*p&&*p!='"'){if(*p=='\\'&&*(p+1))p++;p++;} return *p?p+1:p;}
    if(*p=='['||*p=='{'){
        char open=*p,close=open=='['?']':'}'; int depth=0; p++;
        while(*p){if(*p==open)depth++;else if(*p==close){if(!depth)return p+1;depth--;}
            else if(*p=='"'){p++;while(*p&&*p!='"'){if(*p=='\\'&&*(p+1))p++;p++;}if(*p)p++;continue;}
            p++;}
        return p;
    }
    while(*p&&*p!=','&&*p!='}'&&*p!=']')p++; return p;
}

/* Next quoted string in array; returns pointer past it, or NULL at end */
static const char *js_next_str(const char *p, char *out, size_t sz){
    while(*p&&*p!='"'&&*p!=']')p++;
    if(!*p||*p==']'){if(out)out[0]='\0'; return NULL;}
    p++; size_t i=0;
    while(*p&&*p!='"'&&i<sz-1){if(*p=='\\'&&*(p+1))p++;out[i++]=*p++;}
    if(out)out[i]='\0'; return *p?p+1:p;
}

/* Next object in array; sets *obj_start and *obj_end (inclusive '{' to '}') */
static const char *js_next_obj(const char *p, const char **obj_start, size_t *obj_len){
    while(*p&&*p!='{'&&*p!=']')p++;
    if(!*p||*p==']') return NULL;
    *obj_start=p;
    const char *end=js_skip(p);
    *obj_len=(size_t)(end-p);
    return end;
}

/* ====================================================================
 * SQLite registry
 * ==================================================================== */

static const char *SCHEMA_SQL =
    "PRAGMA journal_mode=WAL;"
    "PRAGMA busy_timeout=5000;"
    "PRAGMA synchronous=NORMAL;"
    "CREATE TABLE IF NOT EXISTS recipes("
    "  code        TEXT PRIMARY KEY,"
    "  name        TEXT NOT NULL,"
    "  version     TEXT NOT NULL DEFAULT '1.0.0',"
    "  description TEXT NOT NULL DEFAULT '',"
    "  hash        TEXT NOT NULL,"
    "  json_text   TEXT NOT NULL,"
    "  source      TEXT NOT NULL DEFAULT 'custom',"
    "  created_at  TEXT NOT NULL,"
    "  updated_at  TEXT NOT NULL"
    ");"
    "CREATE VIRTUAL TABLE IF NOT EXISTS recipes_fts USING fts5("
    "  code, name, description,"
    "  content=recipes, content_rowid=rowid"
    ");"
    "CREATE TRIGGER IF NOT EXISTS recipes_ai AFTER INSERT ON recipes BEGIN"
    "  INSERT INTO recipes_fts(rowid,code,name,description)"
    "  VALUES(new.rowid,new.code,new.name,new.description); END;"
    "CREATE TRIGGER IF NOT EXISTS recipes_ad AFTER DELETE ON recipes BEGIN"
    "  INSERT INTO recipes_fts(recipes_fts,rowid,code,name,description)"
    "  VALUES('delete',old.rowid,old.code,old.name,old.description); END;";

static const char *default_db_path(void){
    static char path[PATH_MAX];
    const char *env=getenv(DB_ENV);
    if(env&&env[0]) return env;
    const char *home=getenv("HOME"); if(!home) home="/tmp";
    snprintf(path,sizeof(path),"%s%s",home,DB_SUBPATH);
    return path;
}

static int ensure_parent(const char *path){
    char tmp[PATH_MAX]; snprintf(tmp,sizeof(tmp),"%s",path);
    char *p=strrchr(tmp,'/'); if(!p) return 0;
    *p='\0';
    for(char *q=tmp+1;*q;q++){
        if(*q=='/'){*q='\0'; mkdir(tmp,0755); *q='/';}
    }
    return mkdir(tmp,0755)==0||errno==EEXIST ? 0 : -1;
}

static sqlite3 *db_open(const char *path){
    ensure_parent(path);
    sqlite3 *db=NULL;
    if(sqlite3_open(path,&db)!=SQLITE_OK){
        fprintf(stderr,"recipe-db: cannot open %s: %s\n",path,sqlite3_errmsg(db));
        sqlite3_close(db); return NULL;
    }
    char *err=NULL;
    if(sqlite3_exec(db,SCHEMA_SQL,NULL,NULL,&err)!=SQLITE_OK){
        fprintf(stderr,"recipe-db: schema error: %s\n",err);
        sqlite3_free(err); sqlite3_close(db); return NULL;
    }
    return db;
}

static void iso_now(char *buf, size_t sz){
    time_t t=time(NULL); struct tm tm;
    gmtime_r(&t,&tm); strftime(buf,sz,"%Y-%m-%dT%H:%M:%SZ",&tm);
}

/* Upsert a recipe row */
static int db_upsert(sqlite3 *db, const char *code, const char *name,
                     const char *version, const char *description,
                     const char *hash, const char *json, const char *source){
    char ts[32]; iso_now(ts,sizeof(ts));
    const char *sql=
        "INSERT INTO recipes(code,name,version,description,hash,json_text,source,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(code) DO UPDATE SET"
        "  name=excluded.name, version=excluded.version,"
        "  description=excluded.description, hash=excluded.hash,"
        "  json_text=excluded.json_text, source=excluded.source,"
        "  updated_at=excluded.updated_at;";
    sqlite3_stmt *stmt; int rc;
    if((rc=sqlite3_prepare_v2(db,sql,-1,&stmt,NULL))!=SQLITE_OK) return rc;
    sqlite3_bind_text(stmt,1,code,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,2,name,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,3,version,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,4,description,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,5,hash,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,6,json,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,7,source,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,8,ts,-1,SQLITE_STATIC);
    sqlite3_bind_text(stmt,9,ts,-1,SQLITE_STATIC);
    rc=sqlite3_step(stmt); sqlite3_finalize(stmt);
    return rc==SQLITE_DONE?SQLITE_OK:rc;
}

/* ====================================================================
 * Built-in recipes
 * ==================================================================== */

static const char *BUILTIN_A1 =
    "{\"code\":\"A1\",\"name\":\"audio to brief\",\"version\":\"1.0.0\","
    "\"description\":\"Transcribe audio and produce a structured brief. "
    "Foundation of every audio pipeline.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],"
     "\"depends_on\":[]},"
    "{\"id\":\"transcribe\",\"bin\":\"bonfyre-transcribe\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/transcribe\"],"
     "\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"clean\",\"bin\":\"bonfyre-transcript-clean\","
     "\"args\":[\"{out}/transcribe\",\"--out\",\"{out}/clean\"],"
     "\"depends_on\":[\"transcribe\"]},"
    "{\"id\":\"brief\",\"bin\":\"bonfyre-brief\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/brief\"],"
     "\"depends_on\":[\"clean\"]}"
    "]}";

static const char *BUILTIN_A2 =
    "{\"code\":\"A2\",\"name\":\"audio to searchable archive\",\"version\":\"1.0.0\","
    "\"description\":\"A1 extended to a LMDB-backed hybrid BM25+cosine searchable archive.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"transcribe\",\"bin\":\"bonfyre-transcribe\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/transcribe\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"clean\",\"bin\":\"bonfyre-transcript-clean\","
     "\"args\":[\"{out}/transcribe\",\"--out\",\"{out}/clean\"],\"depends_on\":[\"transcribe\"]},"
    "{\"id\":\"tag\",\"bin\":\"bonfyre-tag\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/tag\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"brief\",\"bin\":\"bonfyre-brief\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/brief\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"embed\",\"bin\":\"bonfyre-embed\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/embed\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"index\",\"bin\":\"bonfyre-index\","
     "\"args\":[\"{out}/embed\",\"--out\",\"{out}/index\"],\"depends_on\":[\"embed\"]}"
    "]}";

/* A3: full pipeline with parallel stages (tag+tone+brief run concurrently after clean) */
static const char *BUILTIN_A3 =
    "{\"code\":\"A3\",\"name\":\"audio to full pipeline\",\"version\":\"1.0.0\","
    "\"description\":\"23-stage full pipeline: ingest through ledger+meter. "
    "Parallel execution: tag, tone, brief run concurrently after clean.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"media-prep\",\"bin\":\"bonfyre-media-prep\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/media-prep\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"transcribe\",\"bin\":\"bonfyre-transcribe\","
     "\"args\":[\"{out}/media-prep\",\"--out\",\"{out}/transcribe\"],\"depends_on\":[\"media-prep\"]},"
    "{\"id\":\"clean\",\"bin\":\"bonfyre-transcript-clean\","
     "\"args\":[\"{out}/transcribe\",\"--out\",\"{out}/clean\"],\"depends_on\":[\"transcribe\"]},"
    "{\"id\":\"paragraph\",\"bin\":\"bonfyre-paragraph\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/paragraph\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"tone\",\"bin\":\"bonfyre-tone\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/tone\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"tag\",\"bin\":\"bonfyre-tag\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/tag\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"brief\",\"bin\":\"bonfyre-brief\","
     "\"args\":[\"{out}/paragraph\",\"--out\",\"{out}/brief\"],\"depends_on\":[\"paragraph\"]},"
    "{\"id\":\"embed\",\"bin\":\"bonfyre-embed\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/embed\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"render\",\"bin\":\"bonfyre-render\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/render\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"hash\",\"bin\":\"bonfyre-hash\","
     "\"args\":[\"merkle\",\"{out}/render/artifact.json\",\"--out\",\"{out}/hash\"],\"depends_on\":[\"render\"]},"
    "{\"id\":\"proof\",\"bin\":\"bonfyre-proof\","
     "\"args\":[\"{out}/hash\",\"--out\",\"{out}/proof\"],\"depends_on\":[\"hash\"]},"
    "{\"id\":\"offer\",\"bin\":\"bonfyre-offer\","
     "\"args\":[\"{out}/brief\",\"--tone\",\"{out}/tone\",\"--out\",\"{out}/offer\"],"
     "\"depends_on\":[\"brief\",\"tone\"]},"
    "{\"id\":\"pack\",\"bin\":\"bonfyre-pack\","
     "\"args\":[\"{out}/proof\",\"--offer\",\"{out}/offer\",\"--render\",\"{out}/render\","
     "\"--out\",\"{out}/pack\"],\"depends_on\":[\"proof\",\"offer\",\"render\"]},"
    "{\"id\":\"repurpose\",\"bin\":\"bonfyre-repurpose\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/repurpose\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"emit\",\"bin\":\"bonfyre-emit\","
     "\"args\":[\"{out}/repurpose\",\"--out\",\"{out}/emit\"],\"depends_on\":[\"repurpose\"]},"
    "{\"id\":\"compress\",\"bin\":\"bonfyre-compress\","
     "\"args\":[\"{out}/pack\",\"--out\",\"{out}/compress\"],\"depends_on\":[\"pack\"]},"
    "{\"id\":\"index\",\"bin\":\"bonfyre-index\","
     "\"args\":[\"{out}/embed\",\"--out\",\"{out}/index\"],\"depends_on\":[\"embed\"]},"
    "{\"id\":\"stitch\",\"bin\":\"bonfyre-stitch\","
     "\"args\":[\"plan\",\"--out\",\"{out}/stitch\"],\"depends_on\":[\"compress\",\"emit\",\"index\"]},"
    "{\"id\":\"ledger\",\"bin\":\"bonfyre-ledger\","
     "\"args\":[\"{out}/stitch\",\"--out\",\"{out}/ledger\"],\"depends_on\":[\"stitch\"]},"
    "{\"id\":\"meter\",\"bin\":\"bonfyre-meter\","
     "\"args\":[\"record\",\"--out\",\"{out}/meter\"],\"depends_on\":[\"ledger\"]}"
    "]}";

static const char *BUILTIN_M1 =
    "{\"code\":\"M1\",\"name\":\"media to podcast\",\"version\":\"1.0.0\","
    "\"description\":\"Full podcast production: ingest through repurpose+emit+offer+distribute.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"media-prep\",\"bin\":\"bonfyre-media-prep\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/media-prep\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"transcribe\",\"bin\":\"bonfyre-transcribe\","
     "\"args\":[\"{out}/media-prep\",\"--out\",\"{out}/transcribe\"],\"depends_on\":[\"media-prep\"]},"
    "{\"id\":\"clean\",\"bin\":\"bonfyre-transcript-clean\","
     "\"args\":[\"{out}/transcribe\",\"--out\",\"{out}/clean\"],\"depends_on\":[\"transcribe\"]},"
    "{\"id\":\"brief\",\"bin\":\"bonfyre-brief\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/brief\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"clips\",\"bin\":\"bonfyre-clips\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/clips\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"repurpose\",\"bin\":\"bonfyre-repurpose\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/repurpose\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"offer\",\"bin\":\"bonfyre-offer\","
     "\"args\":[\"{out}/brief\",\"--out\",\"{out}/offer\"],\"depends_on\":[\"brief\"]},"
    "{\"id\":\"emit\",\"bin\":\"bonfyre-emit\","
     "\"args\":[\"{out}/repurpose\",\"--out\",\"{out}/emit\"],\"depends_on\":[\"repurpose\"]},"
    "{\"id\":\"distribute\",\"bin\":\"bonfyre-distribute\","
     "\"args\":[\"{out}/emit\",\"--out\",\"{out}/distribute\"],\"depends_on\":[\"emit\"]}"
    "]}";

static const char *BUILTIN_P1 =
    "{\"code\":\"P1\",\"name\":\"proof bundle\",\"version\":\"1.0.0\","
    "\"description\":\"Content-address + BLAKE2b hash + Ed25519 sign + pack + ledger. "
    "Use for any deliverable requiring verifiable provenance.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"hash\",\"bin\":\"bonfyre-hash\","
     "\"args\":[\"file\",\"{input}\",\"--out\",\"{out}/hash\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"proof\",\"bin\":\"bonfyre-proof\","
     "\"args\":[\"{out}/hash\",\"--out\",\"{out}/proof\"],\"depends_on\":[\"hash\"]},"
    "{\"id\":\"pack\",\"bin\":\"bonfyre-pack\","
     "\"args\":[\"{out}/proof\",\"--out\",\"{out}/pack\"],\"depends_on\":[\"proof\"]},"
    "{\"id\":\"ledger\",\"bin\":\"bonfyre-ledger\","
     "\"args\":[\"{out}/pack\",\"--out\",\"{out}/ledger\"],\"depends_on\":[\"pack\"]}"
    "]}";

static const char *BUILTIN_P2 =
    "{\"code\":\"P2\",\"name\":\"verified publish\",\"version\":\"1.0.0\","
    "\"description\":\"P1 (proof bundle) extended with multi-format emit + sync to remote targets.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"hash\",\"bin\":\"bonfyre-hash\","
     "\"args\":[\"file\",\"{input}\",\"--out\",\"{out}/hash\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"proof\",\"bin\":\"bonfyre-proof\","
     "\"args\":[\"{out}/hash\",\"--out\",\"{out}/proof\"],\"depends_on\":[\"hash\"]},"
    "{\"id\":\"pack\",\"bin\":\"bonfyre-pack\","
     "\"args\":[\"{out}/proof\",\"--out\",\"{out}/pack\"],\"depends_on\":[\"proof\"]},"
    "{\"id\":\"ledger\",\"bin\":\"bonfyre-ledger\","
     "\"args\":[\"{out}/pack\",\"--out\",\"{out}/ledger\"],\"depends_on\":[\"pack\"]},"
    "{\"id\":\"emit\",\"bin\":\"bonfyre-emit\","
     "\"args\":[\"{out}/pack\",\"--out\",\"{out}/emit\"],\"depends_on\":[\"pack\"]},"
    "{\"id\":\"sync\",\"bin\":\"bonfyre-sync\","
     "\"args\":[\"{out}/emit\",\"--out\",\"{out}/sync\"],\"depends_on\":[\"emit\"]}"
    "]}";

static const char *BUILTIN_V1 =
    "{\"code\":\"V1\",\"name\":\"media to verified brief\",\"version\":\"1.0.0\","
    "\"description\":\"A1 (audio to brief) + sign + pack + emit. "
    "Produces a verified, multi-format brief from any audio.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"transcribe\",\"bin\":\"bonfyre-transcribe\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/transcribe\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"clean\",\"bin\":\"bonfyre-transcript-clean\","
     "\"args\":[\"{out}/transcribe\",\"--out\",\"{out}/clean\"],\"depends_on\":[\"transcribe\"]},"
    "{\"id\":\"brief\",\"bin\":\"bonfyre-brief\","
     "\"args\":[\"{out}/clean\",\"--out\",\"{out}/brief\"],\"depends_on\":[\"clean\"]},"
    "{\"id\":\"hash\",\"bin\":\"bonfyre-hash\","
     "\"args\":[\"merkle\",\"{out}/brief/artifact.json\",\"--out\",\"{out}/hash\"],"
     "\"depends_on\":[\"brief\"]},"
    "{\"id\":\"proof\",\"bin\":\"bonfyre-proof\","
     "\"args\":[\"{out}/hash\",\"--out\",\"{out}/proof\"],\"depends_on\":[\"hash\"]},"
    "{\"id\":\"pack\",\"bin\":\"bonfyre-pack\","
     "\"args\":[\"{out}/proof\",\"--render\",\"{out}/brief\",\"--out\",\"{out}/pack\"],"
     "\"depends_on\":[\"proof\"]},"
    "{\"id\":\"emit\",\"bin\":\"bonfyre-emit\","
     "\"args\":[\"{out}/pack\",\"--out\",\"{out}/emit\"],\"depends_on\":[\"pack\"]}"
    "]}";

static const char *BUILTIN_R1 =
    "{\"code\":\"R1\",\"name\":\"repo explain\",\"version\":\"1.0.0\","
    "\"description\":\"Source tree to navigable onboarding guide: module map, call graph, "
    "API surface, walkthrough. canon and graph run in parallel after ingest.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"graph\",\"bin\":\"bonfyre-graph\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/graph\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"canon\",\"bin\":\"bonfyre-canon\","
     "\"args\":[\"{out}/ingest\",\"--out\",\"{out}/canon\"],\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"render\",\"bin\":\"bonfyre-render\","
     "\"args\":[\"{out}/graph\",\"--canon\",\"{out}/canon\",\"--out\",\"{out}/render\"],"
     "\"depends_on\":[\"graph\",\"canon\"]},"
    "{\"id\":\"cms\",\"bin\":\"bonfyre-cms\","
     "\"args\":[\"serve\",\"--content\",\"{out}/render\",\"--out\",\"{out}/cms\"],"
     "\"depends_on\":[\"render\"]}"
    "]}";

static const char *BUILTIN_L1 =
    "{\"code\":\"L1\",\"name\":\"revenue track\",\"version\":\"1.0.0\","
    "\"description\":\"Tag a usage event, record it in the ledger, compute margin via finance.\","
    "\"stages\":["
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[]},"
    "{\"id\":\"meter\",\"bin\":\"bonfyre-meter\","
     "\"args\":[\"record\",\"--src\",\"{out}/ingest\",\"--out\",\"{out}/meter\"],"
     "\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"ledger\",\"bin\":\"bonfyre-ledger\","
     "\"args\":[\"{out}/meter\",\"--out\",\"{out}/ledger\"],\"depends_on\":[\"meter\"]},"
    "{\"id\":\"finance\",\"bin\":\"bonfyre-finance\","
     "\"args\":[\"{out}/ledger\",\"--out\",\"{out}/finance\"],\"depends_on\":[\"ledger\"]}"
    "]}";

static const char *BUILTIN_G1 =
    "{\"code\":\"G1\",\"name\":\"api gateway check\",\"version\":\"1.0.0\","
    "\"description\":\"Gate license check + ingest + meter + ledger. "
    "Test harness for the access control stack.\","
    "\"stages\":["
    "{\"id\":\"gate\",\"bin\":\"bonfyre-gate\","
     "\"args\":[\"check\",\"--out\",\"{out}/gate\"],\"depends_on\":[]},"
    "{\"id\":\"ingest\",\"bin\":\"bonfyre-ingest\","
     "\"args\":[\"{input}\",\"--out\",\"{out}/ingest\"],\"depends_on\":[\"gate\"]},"
    "{\"id\":\"meter\",\"bin\":\"bonfyre-meter\","
     "\"args\":[\"record\",\"--src\",\"{out}/ingest\",\"--out\",\"{out}/meter\"],"
     "\"depends_on\":[\"ingest\"]},"
    "{\"id\":\"ledger\",\"bin\":\"bonfyre-ledger\","
     "\"args\":[\"{out}/meter\",\"--out\",\"{out}/ledger\"],\"depends_on\":[\"meter\"]}"
    "]}";

/* Master table */
typedef struct { const char *code; const char *json; } BuiltinEntry;
static BuiltinEntry BUILTINS[] = {
    {"A1", NULL}, /* filled below via pointer aliasing */
    {"A2", NULL},
    {"A3", NULL},
    {"M1", NULL},
    {"P1", NULL},
    {"P2", NULL},
    {"V1", NULL},
    {"R1", NULL},
    {"L1", NULL},
    {"G1", NULL},
    {NULL, NULL}
};
static void init_builtins(void){
    BUILTINS[0].json=BUILTIN_A1;
    BUILTINS[1].json=BUILTIN_A2;
    BUILTINS[2].json=BUILTIN_A3;
    BUILTINS[3].json=BUILTIN_M1;
    BUILTINS[4].json=BUILTIN_P1;
    BUILTINS[5].json=BUILTIN_P2;
    BUILTINS[6].json=BUILTIN_V1;
    BUILTINS[7].json=BUILTIN_R1;
    BUILTINS[8].json=BUILTIN_L1;
    BUILTINS[9].json=BUILTIN_G1;
}

/* Lookup a built-in by code */
static const char *builtin_find(const char *code){
    for(int i=0;BUILTINS[i].code;i++)
        if(strcmp(BUILTINS[i].code,code)==0) return BUILTINS[i].json;
    return NULL;
}

/* ====================================================================
 * Commands
 * ==================================================================== */

static int cmd_help(void){
    printf("bonfyre-recipe %s — pipeline recipe registry\n\n",VERSION);
    printf("Commands:\n");
    printf("  list                  list all recipes (built-in + custom)\n");
    printf("  show <code>           print full recipe JSON\n");
    printf("  add <file.json>       register a custom recipe from file\n");
    printf("  add --builtin         seed registry with all built-in recipes\n");
    printf("  search <query>        full-text search over names + descriptions\n");
    printf("  validate <code>       check all stage binaries exist on PATH\n");
    printf("  hash <code>           print SHA-256 of recipe JSON\n");
    printf("  rm <code>             remove a recipe\n");
    printf("  init [dir]            write starter recipe.json to dir (default: .)\n");
    printf("  status                registry stats\n");
    printf("  help                  this message\n\n");
    printf("Built-in codes: A1 A2 A3 M1 P1 P2 V1 R1 L1 G1\n");
    printf("DB path: %s  (override: $%s)\n",default_db_path(),DB_ENV);
    return 0;
}

static int cmd_list(sqlite3 *db){
    /* First print any custom recipes from DB */
    const char *sql=
        "SELECT code,name,version,source,description FROM recipes ORDER BY source DESC,code;";
    sqlite3_stmt *stmt; int rc,found=0;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)==SQLITE_OK){
        while(sqlite3_step(stmt)==SQLITE_ROW){
            const char *code=(const char*)sqlite3_column_text(stmt,0);
            const char *name=(const char*)sqlite3_column_text(stmt,1);
            const char *ver =(const char*)sqlite3_column_text(stmt,2);
            const char *src =(const char*)sqlite3_column_text(stmt,3);
            const char *desc=(const char*)sqlite3_column_text(stmt,4);
            printf("%-6s  %-32s  %-8s  [%s]\n",code,name,ver,src);
            if(desc&&desc[0]) printf("        %s\n",desc);
            found++;
        }
        sqlite3_finalize(stmt);
    }
    /* Always show built-ins that are not already in DB */
    for(int i=0;BUILTINS[i].code;i++){
        /* Check if already printed */
        rc=0;
        const char *chk="SELECT 1 FROM recipes WHERE code=?";
        sqlite3_stmt *s2;
        if(sqlite3_prepare_v2(db,chk,-1,&s2,NULL)==SQLITE_OK){
            sqlite3_bind_text(s2,1,BUILTINS[i].code,-1,SQLITE_STATIC);
            rc=sqlite3_step(s2)==SQLITE_ROW?1:0;
            sqlite3_finalize(s2);
        }
        if(!rc){
            char name[128]={0},ver[32]={0},desc[512]={0};
            js_str(BUILTINS[i].json,"name",name,sizeof(name));
            js_str(BUILTINS[i].json,"version",ver,sizeof(ver));
            js_str(BUILTINS[i].json,"description",desc,sizeof(desc));
            printf("%-6s  %-32s  %-8s  [built-in]\n",BUILTINS[i].code,name,ver);
            if(desc[0]) printf("        %.80s%s\n",desc,strlen(desc)>80?"...":"");
            found++;
        }
    }
    if(!found) printf("No recipes. Run: bonfyre-recipe add --builtin\n");
    return 0;
}

static int cmd_show(sqlite3 *db, const char *code){
    /* Try DB first */
    const char *sql="SELECT json_text FROM recipes WHERE code=?";
    sqlite3_stmt *stmt;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)==SQLITE_OK){
        sqlite3_bind_text(stmt,1,code,-1,SQLITE_STATIC);
        if(sqlite3_step(stmt)==SQLITE_ROW){
            printf("%s\n",(const char*)sqlite3_column_text(stmt,0));
            sqlite3_finalize(stmt); return 0;
        }
        sqlite3_finalize(stmt);
    }
    /* Try built-ins */
    const char *json=builtin_find(code);
    if(json){ printf("%s\n",json); return 0; }
    fprintf(stderr,"recipe: unknown code '%s'\n",code); return 1;
}

static int register_json(sqlite3 *db, const char *json_text, const char *source){
    char code[32]={0},name[128]={0},ver[32]={0},desc[512]={0},hash[HASH_HEX]={0};
    js_str(json_text,"code",code,sizeof(code));
    js_str(json_text,"name",name,sizeof(name));
    js_str(json_text,"version",ver,sizeof(ver));
    js_str(json_text,"description",desc,sizeof(desc));
    if(!code[0]||!name[0]){
        fprintf(stderr,"recipe: JSON missing 'code' or 'name' field\n"); return 1;
    }
    if(!ver[0]) strcpy(ver,"1.0.0");
    sha256_hex(json_text,strlen(json_text),hash);
    int rc=db_upsert(db,code,name,ver,desc,hash,json_text,source);
    if(rc!=SQLITE_OK){ fprintf(stderr,"recipe: db error %d\n",rc); return 1; }
    printf("registered %s (%s) hash=%s\n",code,name,hash);
    return 0;
}

static int cmd_add(sqlite3 *db, int argc, char **argv){
    if(argc<3){ fprintf(stderr,"usage: bonfyre-recipe add <file.json|--builtin>\n"); return 1; }
    if(strcmp(argv[2],"--builtin")==0){
        int ok=0,fail=0;
        for(int i=0;BUILTINS[i].code;i++){
            if(register_json(db,BUILTINS[i].json,"builtin")==0) ok++;
            else fail++;
        }
        printf("seeded %d built-ins (%d failed)\n",ok,fail);
        return fail>0?1:0;
    }
    /* Read file */
    FILE *fp=fopen(argv[2],"rb");
    if(!fp){ fprintf(stderr,"recipe: cannot open %s: %s\n",argv[2],strerror(errno)); return 1; }
    char *buf=malloc(MAX_JSON); if(!buf){ fclose(fp); return 1; }
    size_t n=fread(buf,1,MAX_JSON-1,fp); fclose(fp); buf[n]='\0';
    int rc=register_json(db,buf,"custom");
    free(buf); return rc;
}

static int cmd_search(sqlite3 *db, const char *query){
    const char *sql=
        "SELECT r.code,r.name,r.version,r.source FROM recipes r"
        " JOIN recipes_fts f ON r.rowid=f.rowid"
        " WHERE recipes_fts MATCH ?"
        " ORDER BY r.code LIMIT 20;";
    sqlite3_stmt *stmt; int found=0;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)!=SQLITE_OK){
        fprintf(stderr,"recipe: search unavailable\n"); return 1;
    }
    sqlite3_bind_text(stmt,1,query,-1,SQLITE_STATIC);
    while(sqlite3_step(stmt)==SQLITE_ROW){
        printf("%-6s  %-32s  %-8s  [%s]\n",
            sqlite3_column_text(stmt,0), sqlite3_column_text(stmt,1),
            sqlite3_column_text(stmt,2), sqlite3_column_text(stmt,3));
        found++;
    }
    sqlite3_finalize(stmt);
    if(!found) printf("No results for '%s'\n",query);
    return 0;
}

static int cmd_validate(sqlite3 *db, const char *code){
    /* Load JSON */
    char json[MAX_JSON]={0};
    const char *sql="SELECT json_text FROM recipes WHERE code=?";
    sqlite3_stmt *stmt;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)==SQLITE_OK){
        sqlite3_bind_text(stmt,1,code,-1,SQLITE_STATIC);
        if(sqlite3_step(stmt)==SQLITE_ROW)
            snprintf(json,sizeof(json),"%s",(const char*)sqlite3_column_text(stmt,0));
        sqlite3_finalize(stmt);
    }
    if(!json[0]){
        const char *bj=builtin_find(code);
        if(!bj){ fprintf(stderr,"recipe: unknown code '%s'\n",code); return 1; }
        snprintf(json,sizeof(json),"%s",bj);
    }
    /* Parse stages and check each bin */
    const char *stages_start=js_arr_start(json,"stages");
    if(!stages_start){ fprintf(stderr,"recipe: no stages array\n"); return 1; }
    int ok=0,fail=0;
    const char *p=stages_start+1;
    while(1){
        const char *obj; size_t obj_len;
        const char *end=js_next_obj(p,&obj,&obj_len);
        if(!end) break;
        char obj_buf[4096]; snprintf(obj_buf,sizeof(obj_buf),"%.*s",(int)obj_len,obj);
        char bin[128]={0}; js_str(obj_buf,"bin",bin,sizeof(bin));
        if(bin[0]){
            /* find on PATH */
            char found_path[PATH_MAX]; int on_path=0;
            const char *PATH=getenv("PATH"); char pb[PATH_MAX];
            const char *start=PATH;
            while(start&&*start){
                const char *colon=strchr(start,':');
                size_t len=colon?(size_t)(colon-start):strlen(start);
                snprintf(pb,sizeof(pb),"%.*s/%s",(int)len,start,bin);
                struct stat st;
                if(stat(pb,&st)==0&&(st.st_mode&S_IXUSR)){on_path=1;strncpy(found_path,pb,sizeof(found_path)-1);break;}
                start=colon?colon+1:NULL;
            }
            char id[64]={0}; js_str(obj_buf,"id",id,sizeof(id));
            if(on_path){ printf("  OK  %-24s  %s\n",bin,found_path); ok++; }
            else        { printf("  !!  %-24s  NOT FOUND\n",bin);     fail++; }
        }
        p=end;
    }
    printf("%d/%d binaries found\n",ok,ok+fail);
    return fail>0?1:0;
}

static int cmd_hash(sqlite3 *db, const char *code){
    char json[MAX_JSON]={0};
    const char *sql="SELECT json_text FROM recipes WHERE code=?";
    sqlite3_stmt *stmt;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)==SQLITE_OK){
        sqlite3_bind_text(stmt,1,code,-1,SQLITE_STATIC);
        if(sqlite3_step(stmt)==SQLITE_ROW)
            snprintf(json,sizeof(json),"%s",(const char*)sqlite3_column_text(stmt,0));
        sqlite3_finalize(stmt);
    }
    if(!json[0]){
        const char *bj=builtin_find(code); if(!bj){ fprintf(stderr,"recipe: unknown '%s'\n",code); return 1; }
        snprintf(json,sizeof(json),"%s",bj);
    }
    char hex[HASH_HEX]; sha256_hex(json,strlen(json),hex);
    printf("%s  %s\n",hex,code); return 0;
}

static int cmd_rm(sqlite3 *db, const char *code){
    const char *sql="DELETE FROM recipes WHERE code=?";
    sqlite3_stmt *stmt;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)!=SQLITE_OK) return 1;
    sqlite3_bind_text(stmt,1,code,-1,SQLITE_STATIC);
    sqlite3_step(stmt); sqlite3_finalize(stmt);
    int changed=sqlite3_changes(db);
    if(changed) printf("removed %s\n",code);
    else printf("not found: %s\n",code);
    return 0;
}

static const char *STARTER_RECIPE =
    "{\n"
    "  \"code\": \"MYRECIPE\",\n"
    "  \"name\": \"my custom pipeline\",\n"
    "  \"version\": \"1.0.0\",\n"
    "  \"description\": \"Describe what this pipeline does.\",\n"
    "  \"stages\": [\n"
    "    {\n"
    "      \"id\": \"ingest\",\n"
    "      \"bin\": \"bonfyre-ingest\",\n"
    "      \"args\": [\"{input}\", \"--out\", \"{out}/ingest\"],\n"
    "      \"depends_on\": []\n"
    "    },\n"
    "    {\n"
    "      \"id\": \"transcribe\",\n"
    "      \"bin\": \"bonfyre-transcribe\",\n"
    "      \"args\": [\"{out}/ingest\", \"--out\", \"{out}/transcribe\"],\n"
    "      \"depends_on\": [\"ingest\"]\n"
    "    },\n"
    "    {\n"
    "      \"id\": \"brief\",\n"
    "      \"bin\": \"bonfyre-brief\",\n"
    "      \"args\": [\"{out}/transcribe\", \"--out\", \"{out}/brief\"],\n"
    "      \"depends_on\": [\"transcribe\"]\n"
    "    }\n"
    "  ]\n"
    "}\n";

static int cmd_init(int argc, char **argv){
    const char *dir=(argc>=3)?argv[2]:".";
    char path[PATH_MAX]; snprintf(path,sizeof(path),"%s/recipe.json",dir);
    struct stat st;
    if(stat(path,&st)==0){ fprintf(stderr,"recipe: %s already exists\n",path); return 1; }
    FILE *fp=fopen(path,"w");
    if(!fp){ fprintf(stderr,"recipe: cannot write %s: %s\n",path,strerror(errno)); return 1; }
    fputs(STARTER_RECIPE,fp); fclose(fp);
    printf("wrote %s\nEdit the code, name, and stages, then:\n"
           "  bonfyre-recipe add %s\n",path,path);
    return 0;
}

static int cmd_status(sqlite3 *db){
    const char *sql="SELECT COUNT(*),SUM(CASE WHEN source='builtin' THEN 1 ELSE 0 END),"
                    "SUM(CASE WHEN source='custom' THEN 1 ELSE 0 END) FROM recipes;";
    sqlite3_stmt *stmt;
    if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)!=SQLITE_OK){ return 1; }
    int total=0,bi=0,custom=0;
    if(sqlite3_step(stmt)==SQLITE_ROW){
        total=sqlite3_column_int(stmt,0);
        bi=sqlite3_column_int(stmt,1);
        custom=sqlite3_column_int(stmt,2);
    }
    sqlite3_finalize(stmt);
    printf("Registry: %s\n",default_db_path());
    printf("  built-in (in DB): %d\n",bi);
    printf("  custom:           %d\n",custom);
    printf("  total in DB:      %d\n",total);
    printf("  built-ins (code): %d  (available even without seeding)\n",
           (int)(sizeof(BUILTINS)/sizeof(BUILTINS[0]))-1);
    return 0;
}

/* ====================================================================
 * main
 * ==================================================================== */

int main(int argc, char **argv){
    init_builtins();
    if(argc<2||strcmp(argv[1],"help")==0||strcmp(argv[1],"--help")==0)
        return cmd_help();
    if(strcmp(argv[1],"init")==0) return cmd_init(argc,argv);

    sqlite3 *db=db_open(default_db_path());
    if(!db) return 1;
    int rc=0;

    if(strcmp(argv[1],"list")==0)    rc=cmd_list(db);
    else if(strcmp(argv[1],"show")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-recipe show <code>\n");rc=1;}
        else rc=cmd_show(db,argv[2]);
    }
    else if(strcmp(argv[1],"add")==0)    rc=cmd_add(db,argc,argv);
    else if(strcmp(argv[1],"search")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-recipe search <query>\n");rc=1;}
        else rc=cmd_search(db,argv[2]);
    }
    else if(strcmp(argv[1],"validate")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-recipe validate <code>\n");rc=1;}
        else rc=cmd_validate(db,argv[2]);
    }
    else if(strcmp(argv[1],"hash")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-recipe hash <code>\n");rc=1;}
        else rc=cmd_hash(db,argv[2]);
    }
    else if(strcmp(argv[1],"rm")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-recipe rm <code>\n");rc=1;}
        else rc=cmd_rm(db,argv[2]);
    }
    else if(strcmp(argv[1],"status")==0) rc=cmd_status(db);
    else{
        fprintf(stderr,"bonfyre-recipe: unknown command '%s'\n",argv[1]);
        fprintf(stderr,"Run 'bonfyre-recipe help' for usage.\n"); rc=1;
    }

    sqlite3_close(db);
    return rc;
}
