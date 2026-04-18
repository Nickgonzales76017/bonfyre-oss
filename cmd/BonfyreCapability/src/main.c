/*
 * bonfyre-capability — capability discovery and matching layer.
 *
 * A searchable registry of everything Bonfyre can do: which binary handles
 * which task, what model it uses, what hardware tier it needs, and what it
 * costs. Pipeline authors query this registry to find the right tool without
 * hard-coding binary names or model IDs.
 *
 * DB: ~/.local/share/bonfyre/capability.db  (override: $BONFYRE_CAPABILITY_DB)
 *
 * Commands:
 *   bonfyre-capability status              — registry summary
 *   bonfyre-capability list                — all capabilities
 *   bonfyre-capability search <query>      — full-text capability search
 *   bonfyre-capability show <cap-id>       — full capability record
 *   bonfyre-capability match <description> — find best match for a task
 *   bonfyre-capability register <file.json>— add/update a capability
 *   bonfyre-capability index               — rebuild full-text index
 *   bonfyre-capability help                — this message
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <sqlite3.h>

#define VERSION    "1.0.0"
#define DB_ENV     "BONFYRE_CAPABILITY_DB"
#define DB_SUBPATH "/.local/share/bonfyre/capability.db"

static void db_path(char *buf,size_t len){
    const char *e=getenv(DB_ENV);
    if(e){snprintf(buf,len,"%s",e);return;}
    const char *h=getenv("HOME");if(!h)h="/tmp";
    snprintf(buf,len,"%s%s",h,DB_SUBPATH);
}
static void ensure_dir(const char *p){
    char t[4096];snprintf(t,sizeof(t),"%s",p);
    for(char *q=t+1;*q;q++){if(*q=='/'){*q='\0';mkdir(t,0755);*q='/';}}
}

static const char *SCHEMA=
    "PRAGMA journal_mode=WAL;"
    "CREATE TABLE IF NOT EXISTS capabilities("
    "  id            TEXT PRIMARY KEY,"
    "  name          TEXT NOT NULL,"
    "  description   TEXT NOT NULL,"
    "  tags          TEXT,"      /* comma-separated */
    "  binary        TEXT NOT NULL,"
    "  command       TEXT,"
    "  model_id      TEXT,"
    "  hardware_tier TEXT NOT NULL DEFAULT 'cpu'," /* cpu|gpu|tpu */
    "  cost_estimate REAL NOT NULL DEFAULT 0.0,"  /* USD per call est */
    "  latency_tier  TEXT NOT NULL DEFAULT 'fast',"/* instant|fast|batch|deep */
    "  updated       INTEGER NOT NULL"
    ");"
    "CREATE VIRTUAL TABLE IF NOT EXISTS cap_fts USING fts5("
    "  id,name,description,tags,binary,"
    "  content='capabilities',content_rowid='rowid'"
    ");";

/* Built-in capability seed */
static const struct {
    const char *id,*name,*desc,*tags,*binary,*command,*model,*hw,*lat;
    double cost;
} BUILTINS[]={
    {"audio-ingest","Audio Ingest","Accept audio files and normalize them for pipeline processing","audio,intake,ingest","bonfyre-intake","add","—","cpu","instant",0.001},
    {"speech-to-text","Speech to Text","Transcribe audio to text using Whisper","transcribe,stt,speech,audio","bonfyre-transcribe","run","whisper-large","gpu","batch",0.006},
    {"speaker-diarize","Speaker Diarization","Identify and separate speakers in a transcript","diarize,speaker,segmentation","bonfyre-transcribe","diarize","pyannote","gpu","batch",0.010},
    {"summarize-text","Text Summarization","Summarize long transcripts or documents","summarize,nlp,condensed","bonfyre-brief","run","gpt-4o-mini","cpu","fast",0.002},
    {"entity-resolve","Entity Resolution","Resolve cross-modal observations to identity graph","entity,identity,resolution","bonfyre-entity","resolve","—","cpu","instant",0.000},
    {"translate-text","Text Translation","Translate transcript text to target language","translate,i18n,language","bonfyre-translate","run","gpt-4o","cpu","fast",0.005},
    {"tag-artifact","Artifact Tagging","Tag artifacts with topics, speakers, and metadata","tags,taxonomy,metadata","bonfyre-tag","run","llama-3-8b","cpu","fast",0.001},
    {"quality-eval","Quality Evaluation","Score artifacts with HE-SLI metrics","quality,eval,scoring","bonfyre-control","score","—","cpu","instant",0.000},
    {"cost-route","Cost-Aware Routing","Route inference to cheapest model within budget","economy,budget,routing","bonfyre-economy","route","—","cpu","instant",0.000},
    {"ab-compete","A/B Competition","Run stage variants and score winners","compete,ab,experiment","bonfyre-compete","run","—","cpu","fast",0.000},
    {"flow-define","Flow Definition","Define and run coroutine-native pipeline graphs","flow,pipeline,dag","bonfyre-flow","define","—","cpu","instant",0.000},
    {"memory-space","Shared Memory","Create named in-process shared memory spaces","space,memory,ipc","bonfyre-space","open","—","cpu","instant",0.000},
    {"threshold-tune","Threshold Tuning","Auto-tune stage quality thresholds from feedback","learn,tune,feedback","bonfyre-learn","tune","—","cpu","fast",0.000},
    {"distribute","Distribution","Package and distribute artifacts to cloud/CDN","distribute,cdn,upload","bonfyre-distribute","run","—","cpu","batch",0.010},
    {"batch-queue","Batch Queue","Queue artifacts for background batch processing","queue,batch,async","bonfyre-queue","add","—","cpu","instant",0.000},
};
static const int NBUILTINS=(int)(sizeof(BUILTINS)/sizeof(BUILTINS[0]));

static sqlite3 *open_db(void){
    char path[4096];db_path(path,sizeof(path));ensure_dir(path);
    sqlite3 *db=NULL;
    if(sqlite3_open(path,&db)!=SQLITE_OK){fprintf(stderr,"db error\n");exit(1);}
    char *err=NULL;sqlite3_exec(db,SCHEMA,NULL,NULL,&err);
    if(err){fprintf(stderr,"%s\n",err);sqlite3_free(err);exit(1);}
    /* seed builtins */
    time_t now=time(NULL);
    for(int i=0;i<NBUILTINS;i++){
        sqlite3_stmt *st=NULL;
        sqlite3_prepare_v2(db,
            "INSERT OR IGNORE INTO capabilities"
            "(id,name,description,tags,binary,command,model_id,hardware_tier,cost_estimate,latency_tier,updated)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            -1,&st,NULL);
        sqlite3_bind_text(st,1,BUILTINS[i].id,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,2,BUILTINS[i].name,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,3,BUILTINS[i].desc,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,4,BUILTINS[i].tags,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,5,BUILTINS[i].binary,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,6,BUILTINS[i].command,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,7,BUILTINS[i].model,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,8,BUILTINS[i].hw,-1,SQLITE_STATIC);
        sqlite3_bind_double(st,9,BUILTINS[i].cost);
        sqlite3_bind_text(st,10,BUILTINS[i].lat,-1,SQLITE_STATIC);
        sqlite3_bind_int64(st,11,(sqlite3_int64)now);
        sqlite3_step(st);sqlite3_finalize(st);
    }
    return db;
}

static void rebuild_index(sqlite3 *db){
    char *err=NULL;
    sqlite3_exec(db,"INSERT INTO cap_fts(cap_fts) VALUES('rebuild')",NULL,NULL,&err);
    if(err){sqlite3_free(err);}/* ignore if already populated */
}

static void cmd_status(sqlite3 *db){
    sqlite3_stmt *st=NULL;
    int total=0;
    sqlite3_prepare_v2(db,"SELECT COUNT(*) FROM capabilities",-1,&st,NULL);
    if(sqlite3_step(st)==SQLITE_ROW) total=sqlite3_column_int(st,0);
    sqlite3_finalize(st);
    printf("bonfyre-capability %s\n  capabilities: %d\n",VERSION,total);
    /* breakdown by hardware tier */
    sqlite3_prepare_v2(db,"SELECT hardware_tier,COUNT(*) FROM capabilities GROUP BY hardware_tier ORDER BY hardware_tier",-1,&st,NULL);
    printf("  by hardware:\n");
    while(sqlite3_step(st)==SQLITE_ROW)
        printf("    %-8s %d\n",(const char*)sqlite3_column_text(st,0),sqlite3_column_int(st,1));
    sqlite3_finalize(st);
}

static void cmd_list(sqlite3 *db){
    sqlite3_stmt *st=NULL;
    sqlite3_prepare_v2(db,
        "SELECT id,name,binary,hardware_tier,latency_tier,cost_estimate FROM capabilities ORDER BY name",
        -1,&st,NULL);
    printf("%-25s  %-30s  %-20s  %-5s  %-8s  %10s\n","ID","NAME","BINARY","HW","LATENCY","COST EST");
    while(sqlite3_step(st)==SQLITE_ROW)
        printf("%-25s  %-30s  %-20s  %-5s  %-8s  %10.4f\n",
            (const char*)sqlite3_column_text(st,0),(const char*)sqlite3_column_text(st,1),
            (const char*)sqlite3_column_text(st,2),(const char*)sqlite3_column_text(st,3),
            (const char*)sqlite3_column_text(st,4),sqlite3_column_double(st,5));
    sqlite3_finalize(st);
}

static void cmd_search(sqlite3 *db,const char *q){
    /* attempt FTS, fall back to LIKE */
    sqlite3_stmt *st=NULL;
    if(sqlite3_prepare_v2(db,
        "SELECT c.id,c.name,c.binary,c.description FROM cap_fts f"
        " JOIN capabilities c ON c.id=f.id WHERE cap_fts MATCH ? LIMIT 10",
        -1,&st,NULL)==SQLITE_OK){
        sqlite3_bind_text(st,1,q,-1,SQLITE_STATIC);
    } else {
        char pat[256];snprintf(pat,sizeof(pat),"%%%s%%",q);
        sqlite3_prepare_v2(db,
            "SELECT id,name,binary,description FROM capabilities"
            " WHERE name LIKE ? OR description LIKE ? OR tags LIKE ? LIMIT 10",
            -1,&st,NULL);
        sqlite3_bind_text(st,1,pat,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,2,pat,-1,SQLITE_STATIC);
        sqlite3_bind_text(st,3,pat,-1,SQLITE_STATIC);
    }
    printf("search: '%s'\n%-25s  %-20s  %s\n","—",q,"BINARY","DESCRIPTION");
    printf("%-25s  %-20s  %s\n","ID","BINARY","DESCRIPTION");
    while(sqlite3_step(st)==SQLITE_ROW)
        printf("%-25s  %-20s  %.60s\n",
            (const char*)sqlite3_column_text(st,0),(const char*)sqlite3_column_text(st,2),
            (const char*)sqlite3_column_text(st,3));
    sqlite3_finalize(st);
}

static void cmd_show(sqlite3 *db,const char *cid){
    sqlite3_stmt *st=NULL;
    sqlite3_prepare_v2(db,
        "SELECT id,name,description,tags,binary,command,model_id,hardware_tier,cost_estimate,latency_tier"
        " FROM capabilities WHERE id=?",
        -1,&st,NULL);
    sqlite3_bind_text(st,1,cid,-1,SQLITE_STATIC);
    if(sqlite3_step(st)!=SQLITE_ROW){
        fprintf(stderr,"capability not found: %s\n",cid);sqlite3_finalize(st);return;
    }
    printf("id            : %s\nname          : %s\ndescription   : %s\ntags          : %s\n"
           "binary        : %s\ncommand       : %s\nmodel         : %s\nhardware      : %s\n"
           "cost est      : $%.4f/call\nlatency tier  : %s\n",
        (const char*)sqlite3_column_text(st,0),(const char*)sqlite3_column_text(st,1),
        (const char*)sqlite3_column_text(st,2),(const char*)sqlite3_column_text(st,3),
        (const char*)sqlite3_column_text(st,4),(const char*)sqlite3_column_text(st,5),
        (const char*)sqlite3_column_text(st,6),(const char*)sqlite3_column_text(st,7),
        sqlite3_column_double(st,8),(const char*)sqlite3_column_text(st,9));
    sqlite3_finalize(st);
}

static void cmd_match(sqlite3 *db,const char *desc){
    /* keyword overlap match — tokenize description into terms */
    printf("match: '%s'\n\n",desc);
    char copy[2048];snprintf(copy,sizeof(copy),"%s",desc);
    char pat[2048]=""; char *tok=strtok(copy," \t,.");
    while(tok){
        if(strlen(tok)>3){
            char term[256];snprintf(term,sizeof(term),"%%%s%%",tok);
            /* search each term */
            sqlite3_stmt *st=NULL;
            sqlite3_prepare_v2(db,
                "SELECT id,name,binary,cost_estimate FROM capabilities"
                " WHERE (name LIKE ? OR tags LIKE ? OR description LIKE ?) LIMIT 3",
                -1,&st,NULL);
            sqlite3_bind_text(st,1,term,-1,SQLITE_STATIC);
            sqlite3_bind_text(st,2,term,-1,SQLITE_STATIC);
            sqlite3_bind_text(st,3,term,-1,SQLITE_STATIC);
            while(sqlite3_step(st)==SQLITE_ROW){
                char line[256];snprintf(line,sizeof(line),"%s",
                    (const char*)sqlite3_column_text(st,0));
                /* deduplicate: check if id already printed */
                if(!strstr(pat,line)){
                    printf("  %-25s  %-20s  $%.4f\n",line,
                        (const char*)sqlite3_column_text(st,2),
                        sqlite3_column_double(st,3));
                    strncat(pat,"|",sizeof(pat)-strlen(pat)-1);
                    strncat(pat,line,sizeof(pat)-strlen(pat)-1);
                }
            }
            sqlite3_finalize(st);
        }
        tok=strtok(NULL," \t,.");
    }
    (void)pat;
}

static void cmd_index(sqlite3 *db){
    rebuild_index(db);
    printf("capability index rebuilt\n");
}

static void cmd_help(void){
    printf(
"bonfyre-capability %s — capability discovery and matching\n\n"
"USAGE\n"
"  bonfyre-capability <command> [args]\n\n"
"COMMANDS\n"
"  status              registry summary\n"
"  list                all capabilities\n"
"  search <query>      full-text search\n"
"  show <cap-id>       full capability record\n"
"  match <description> find best capability for a natural-language task\n"
"  register <file.json>  add or update a capability\n"
"  index               rebuild full-text search index\n"
"  help                this message\n\n"
"CAPABILITY FIELDS\n"
"  id · name · description · tags · binary · command · model_id\n"
"  hardware_tier (cpu|gpu|tpu) · cost_estimate (USD/call) · latency_tier\n\n"
"BUILT-IN CAPABILITIES (15)\n"
"  audio-ingest · speech-to-text · speaker-diarize · summarize-text\n"
"  entity-resolve · translate-text · tag-artifact · quality-eval\n"
"  cost-route · ab-compete · flow-define · memory-space · threshold-tune\n"
"  distribute · batch-queue\n\n"
"ENVIRONMENT\n"
"  BONFYRE_CAPABILITY_DB   override DB path\n",
    VERSION);
}

int main(int argc,char **argv){
    if(argc<2||strcmp(argv[1],"help")==0||strcmp(argv[1],"--help")==0){cmd_help();return 0;}
    sqlite3 *db=open_db();
    int rc=0;const char *cmd=argv[1];
    if(strcmp(cmd,"status")==0) cmd_status(db);
    else if(strcmp(cmd,"list")==0) cmd_list(db);
    else if(strcmp(cmd,"search")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-capability search <query>\n");rc=1;}
        else cmd_search(db,argv[2]);
    } else if(strcmp(cmd,"show")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-capability show <cap-id>\n");rc=1;}
        else cmd_show(db,argv[2]);
    } else if(strcmp(cmd,"match")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-capability match <description>\n");rc=1;}
        else cmd_match(db,argv[2]);
    } else if(strcmp(cmd,"index")==0) cmd_index(db);
    else {fprintf(stderr,"bonfyre-capability: unknown command: %s\n",cmd);rc=1;}
    sqlite3_close(db);return rc;
}
