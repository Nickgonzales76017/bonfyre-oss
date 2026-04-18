/*
 * BonfyreRun — pipeline shortcode executor.
 *
 * Resolves a recipe from the registry (bonfyre-recipe's SQLite DB or built-ins),
 * validates stage binaries, executes the stage DAG with level-parallel concurrency,
 * streams progress to stderr, and writes a run-manifest.json on completion.
 *
 * Commands:
 *   bonfyre-run help                      — list all available recipe codes
 *   bonfyre-run show <code>               — print stage chain for a recipe
 *   bonfyre-run validate <code>           — check all stage binaries exist on PATH
 *   bonfyre-run dry-run <code> <input>    — print resolved execution plan, no execution
 *   bonfyre-run init                      — write starter recipe.json to CWD via bonfyre-recipe
 *   bonfyre-run <code> <input> [OPTIONS]  — execute the recipe pipeline
 *
 * Execution options:
 *   --out DIR          output directory  (default: ./bonfyre-out/<timestamp>)
 *   --dry-run          alias for dry-run subcommand
 *   --from-stage ID    skip stages before this one
 *   --to-stage ID      stop after this stage (inclusive)
 *   --resume           reuse --out dir, skip stages whose out dir already exists
 *   --quiet            print only errors + final manifest path
 *   --verbose          print full stage stdout/stderr
 *   --tier TIER        pass --tier flag to every stage that accepts it
 *   --stage-opts S     per-stage extra args: "stageid:--flag val,stageid2:--flag val"
 *   --batch DIR        run against every file in DIR, one subdir per input
 *   --parallel N       max stages to run concurrently per level  (default: 8)
 *   --db FILE          recipe DB path  (default: ~/.local/share/bonfyre/recipes.db)
 */

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <sqlite3.h>

#define VERSION        "1.0.0"
#define MAX_STAGES     32
#define MAX_ARGS       32
#define MAX_ARG_LEN    512
#define MAX_DEPS       16
#define MAX_ID         64
#define MAX_JSON       65536
#define DB_ENV         "BONFYRE_RECIPE_DB"
#define DB_SUBPATH     "/.local/share/bonfyre/recipes.db"

/* ====================================================================
 * Stage + Recipe structures
 * ==================================================================== */

typedef struct {
    char   id[MAX_ID];
    char   bin[256];
    char   args[MAX_ARGS][MAX_ARG_LEN];
    int    n_args;
    char   deps[MAX_DEPS][MAX_ID];
    int    n_deps;
    int    level;          /* topological level — same level = can run in parallel */
    int    status;         /* 0=pending 1=running 2=ok 3=failed 4=skipped */
    int    exit_code;
    double wall_ms;
} Stage;

typedef struct {
    char   code[32];
    char   name[128];
    char   version[32];
    char   description[512];
    char   hash[65];       /* SHA-256 of json_text */
    Stage  stages[MAX_STAGES];
    int    n_stages;
} Recipe;

typedef struct {
    char   input[PATH_MAX];
    char   out[PATH_MAX];
    char   from_stage[MAX_ID];
    char   to_stage[MAX_ID];
    char   tier[32];
    char   stage_opts[2048];  /* "id:--flag val,id2:--flag2 val2" */
    char   db_path[PATH_MAX];
    int    dry_run;
    int    resume;
    int    quiet;
    int    verbose;
    int    max_parallel;
    int    batch;
    char   batch_dir[PATH_MAX];
} RunOpts;

/* ====================================================================
 * Timing
 * ==================================================================== */

static double mono_ms(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC,&ts);
    return ts.tv_sec*1000.0+ts.tv_nsec/1e6;
}

static void iso_now(char *buf, size_t sz){
    time_t t=time(NULL); struct tm tm;
    gmtime_r(&t,&tm); strftime(buf,sz,"%Y-%m-%dT%H:%M:%SZ",&tm);
}

/* ====================================================================
 * Minimal JSON helpers (same approach as bonfyre-recipe)
 * ==================================================================== */

static const char *js_ws(const char *p){ while(*p==' '||*p=='\t'||*p=='\n'||*p=='\r')p++; return p; }

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

static const char *js_arr_start(const char *json, const char *key){
    char needle[128]; snprintf(needle,sizeof(needle),"\"%s\"",key);
    const char *p=strstr(json,needle); if(!p) return NULL;
    p+=strlen(needle); p=js_ws(p); if(*p!=':'||!(p=js_ws(p+1))||*p!='[') return NULL;
    return p;
}

static const char *js_skip(const char *p){
    p=js_ws(p);
    if(*p=='"'){p++;while(*p&&*p!='"'){if(*p=='\\'&&*(p+1))p++;p++;}return *p?p+1:p;}
    if(*p=='['||*p=='{'){
        char open=*p,close=open=='['?']':'}'; int depth=0; p++;
        while(*p){
            if(*p==open)depth++;
            else if(*p==close){if(!depth)return p+1;depth--;}
            else if(*p=='"'){p++;while(*p&&*p!='"'){if(*p=='\\'&&*(p+1))p++;p++;}if(*p)p++;continue;}
            p++;
        }
        return p;
    }
    while(*p&&*p!=','&&*p!='}'&&*p!=']')p++; return p;
}

static const char *js_next_str_in_arr(const char *p, char *out, size_t sz){
    while(*p&&*p!='"'&&*p!=']')p++;
    if(!*p||*p==']'){if(out)out[0]='\0';return NULL;}
    p++; size_t i=0;
    while(*p&&*p!='"'&&i<sz-1){if(*p=='\\'&&*(p+1))p++;out[i++]=*p++;}
    if(out)out[i]='\0'; return *p?p+1:p;
}

static const char *js_next_obj(const char *p, const char **obj_start, size_t *obj_len){
    while(*p&&*p!='{'&&*p!=']')p++;
    if(!*p||*p==']') return NULL;
    *obj_start=p;
    const char *end=js_skip(p);
    *obj_len=(size_t)(end-p);
    return end;
}

/* ====================================================================
 * Recipe loading
 * ==================================================================== */

static const char *default_db_path(void){
    static char path[PATH_MAX];
    const char *env=getenv(DB_ENV);
    if(env&&env[0]) return env;
    const char *home=getenv("HOME"); if(!home) home="/tmp";
    snprintf(path,sizeof(path),"%s%s",home,DB_SUBPATH);
    return path;
}

/* Load recipe JSON by code from SQLite or built-ins (via bonfyre-recipe show <code>) */
static int load_recipe_json(const char *code, const char *db_path,
                             char *json_out, size_t json_sz)
{
    /* 1. Try SQLite directly */
    sqlite3 *db=NULL;
    if(sqlite3_open(db_path,&db)==SQLITE_OK){
        const char *sql="SELECT json_text FROM recipes WHERE code=?";
        sqlite3_stmt *stmt;
        if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)==SQLITE_OK){
            sqlite3_bind_text(stmt,1,code,-1,SQLITE_STATIC);
            if(sqlite3_step(stmt)==SQLITE_ROW){
                snprintf(json_out,json_sz,"%s",(const char*)sqlite3_column_text(stmt,0));
                sqlite3_finalize(stmt); sqlite3_close(db); return 0;
            }
            sqlite3_finalize(stmt);
        }
        sqlite3_close(db);
    }

    /* 2. Fall back: spawn bonfyre-recipe show <code> */
    char cmd[PATH_MAX+64];
    snprintf(cmd,sizeof(cmd),"bonfyre-recipe show '%s' 2>/dev/null",code);
    FILE *fp=popen(cmd,"r");
    if(fp){
        size_t n=fread(json_out,1,json_sz-1,fp);
        pclose(fp);
        if(n>0&&json_out[0]=='{'){json_out[n]='\0';return 0;}
    }

    return -1;  /* not found */
}

static int parse_stages(const char *json, Stage *stages, int max_stages){
    const char *arr=js_arr_start(json,"stages");
    if(!arr) return 0;
    int n=0;
    const char *p=arr+1;
    while(n<max_stages){
        const char *obj; size_t obj_len;
        const char *end=js_next_obj(p,&obj,&obj_len);
        if(!end) break;

        /* copy object into a local buffer so js_str can scan it safely */
        char buf[4096];
        size_t bsz=obj_len<sizeof(buf)-1?obj_len:sizeof(buf)-1;
        memcpy(buf,obj,bsz); buf[bsz]='\0';

        js_str(buf,"id",  stages[n].id,  sizeof(stages[n].id));
        js_str(buf,"bin", stages[n].bin, sizeof(stages[n].bin));

        /* args array */
        const char *aa=js_arr_start(buf,"args");
        if(aa){
            const char *ap=aa+1; stages[n].n_args=0;
            while(stages[n].n_args<MAX_ARGS){
                char arg[MAX_ARG_LEN];
                ap=js_next_str_in_arr(ap,arg,sizeof(arg));
                if(!ap) break;
                strncpy(stages[n].args[stages[n].n_args++],arg,MAX_ARG_LEN-1);
            }
        }

        /* depends_on array */
        const char *da=js_arr_start(buf,"depends_on");
        if(da){
            const char *dp=da+1; stages[n].n_deps=0;
            while(stages[n].n_deps<MAX_DEPS){
                char dep[MAX_ID];
                dp=js_next_str_in_arr(dp,dep,sizeof(dep));
                if(!dp||!dep[0]) break;
                strncpy(stages[n].deps[stages[n].n_deps++],dep,MAX_ID-1);
            }
        }

        stages[n].level=0; stages[n].status=0; stages[n].exit_code=-1; stages[n].wall_ms=0;
        p=end; n++;
    }
    return n;
}

static int load_recipe(const char *code, const char *db_path, Recipe *r){
    char json[MAX_JSON]={0};
    if(load_recipe_json(code,db_path,json,sizeof(json))!=0){
        fprintf(stderr,"run: recipe '%s' not found. Try: bonfyre-recipe list\n",code);
        return 1;
    }
    memset(r,0,sizeof(*r));
    strncpy(r->code,code,sizeof(r->code)-1);
    js_str(json,"name",       r->name,        sizeof(r->name));
    js_str(json,"version",    r->version,     sizeof(r->version));
    js_str(json,"description",r->description, sizeof(r->description));
    r->n_stages=parse_stages(json,r->stages,MAX_STAGES);
    if(r->n_stages==0){
        fprintf(stderr,"run: recipe '%s' has no stages\n",code); return 1;
    }
    return 0;
}

/* ====================================================================
 * Topological level assignment
 * Works correctly for DAGs including parallel (fan-out) branches.
 * ================================================================== */

static int stage_idx(const Recipe *r, const char *id){
    for(int i=0;i<r->n_stages;i++)
        if(strcmp(r->stages[i].id,id)==0) return i;
    return -1;
}

static void assign_levels(Recipe *r){
    /* Iteratively compute level = max(dep levels) + 1 until stable */
    int changed=1;
    while(changed){
        changed=0;
        for(int i=0;i<r->n_stages;i++){
            for(int d=0;d<r->stages[i].n_deps;d++){
                int di=stage_idx(r,r->stages[i].deps[d]);
                if(di>=0&&r->stages[di].level+1>r->stages[i].level){
                    r->stages[i].level=r->stages[di].level+1;
                    changed=1;
                }
            }
        }
    }
}

/* ====================================================================
 * Variable substitution
 * Tokens: {input} → input path,  {out} → base out dir,
 *         {out}/STAGE_ID is just a literal path — no extra substitution needed.
 * ==================================================================== */

static void subst(const char *template, const char *input,
                  const char *out_dir, char *result, size_t sz)
{
    char tmp[MAX_ARG_LEN*2]={0};
    const char *p=template;
    size_t i=0;
    while(*p&&i<sizeof(tmp)-1){
        if(strncmp(p,"{input}",7)==0){
            size_t l=strlen(input); if(i+l>=sizeof(tmp)-1) break;
            memcpy(tmp+i,input,l); i+=l; p+=7;
        } else if(strncmp(p,"{out}",5)==0){
            size_t l=strlen(out_dir); if(i+l>=sizeof(tmp)-1) break;
            memcpy(tmp+i,out_dir,l); i+=l; p+=5;
        } else {
            tmp[i++]=*p++;
        }
    }
    tmp[i]='\0';
    strncpy(result,tmp,sz-1); result[sz-1]='\0';
}

/* ====================================================================
 * PATH lookup
 * ==================================================================== */

static int on_path(const char *bin){
    const char *PATH=getenv("PATH"); if(!PATH) return 0;
    char pb[PATH_MAX];
    const char *s=PATH;
    while(s&&*s){
        const char *c=strchr(s,':');
        size_t len=c?(size_t)(c-s):strlen(s);
        snprintf(pb,sizeof(pb),"%.*s/%s",(int)len,s,bin);
        struct stat st;
        if(stat(pb,&st)==0&&(st.st_mode&S_IXUSR)) return 1;
        s=c?c+1:NULL;
    }
    return 0;
}

/* ====================================================================
 * Stage execution — fork+exec, capture exit code
 * ==================================================================== */

typedef struct {
    int    stage_idx;
    Stage *stage;
    RunOpts *opts;
    char   resolved_args[MAX_ARGS][MAX_ARG_LEN];
    int    n_args;
} StageJob;

static int dir_exists(const char *path){
    struct stat st; return stat(path,&st)==0&&S_ISDIR(st.st_mode);
}

/* Build fully-resolved arg vector for a stage */
static int build_argv(const Stage *s, const RunOpts *opts,
                      char resolved[MAX_ARGS][MAX_ARG_LEN], int *n_out)
{
    *n_out=0;
    resolved[(*n_out)][0]='\0';
    strncpy(resolved[(*n_out)++], s->bin, MAX_ARG_LEN-1);  /* argv[0] = binary name */

    for(int i=0;i<s->n_args;i++){
        char tmp[MAX_ARG_LEN];
        subst(s->args[i], opts->input, opts->out, tmp, sizeof(tmp));
        strncpy(resolved[(*n_out)++], tmp, MAX_ARG_LEN-1);
        if(*n_out>=MAX_ARGS-4) break;
    }

    /* Inject --tier if set */
    if(opts->tier[0]){
        strncpy(resolved[(*n_out)++], "--tier",    MAX_ARG_LEN-1);
        strncpy(resolved[(*n_out)++], opts->tier,  MAX_ARG_LEN-1);
    }

    /* Per-stage extra opts: "stageid:--flag val,..." */
    if(opts->stage_opts[0]){
        /* scan for "ID:..." segments */
        char tmp2[2048]; strncpy(tmp2,opts->stage_opts,sizeof(tmp2)-1);
        char *seg=strtok(tmp2,",");
        while(seg){
            char *colon=strchr(seg,':');
            if(colon){
                *colon='\0';
                if(strcmp(seg,s->id)==0){
                    /* split remaining on spaces */
                    char *flag=strtok(colon+1," ");
                    while(flag&&*n_out<MAX_ARGS-1){
                        strncpy(resolved[(*n_out)++],flag,MAX_ARG_LEN-1);
                        flag=strtok(NULL," ");
                    }
                }
            }
            seg=strtok(NULL,",");
        }
    }
    return 0;
}

/* ====================================================================
 * DAG execution — level-by-level with parallelism within each level
 * ==================================================================== */

static int stages_run(Recipe *r, RunOpts *opts){
    /* Determine level range (from_stage / to_stage support) */
    int from_level=0, to_level=INT32_MAX;
    if(opts->from_stage[0]){
        int fi=stage_idx(r,opts->from_stage);
        if(fi>=0) from_level=r->stages[fi].level;
    }
    if(opts->to_stage[0]){
        int ti=stage_idx(r,opts->to_stage);
        if(ti>=0) to_level=r->stages[ti].level;
    }

    /* Find max level */
    int max_level=0;
    for(int i=0;i<r->n_stages;i++)
        if(r->stages[i].level>max_level) max_level=r->stages[i].level;

    int any_failed=0;

    for(int lvl=0;lvl<=max_level&&!any_failed;lvl++){
        /* Collect stages at this level that need execution */
        int idxs[MAX_STAGES]; int n_at_level=0;
        for(int i=0;i<r->n_stages;i++){
            if(r->stages[i].level!=lvl) continue;
            /* --from/--to-stage range */
            if(lvl<from_level){ r->stages[i].status=4; continue; }
            if(lvl>to_level)  { r->stages[i].status=4; continue; }
            /* --resume: skip if output dir already exists */
            if(opts->resume){
                char out_dir[PATH_MAX];
                snprintf(out_dir,sizeof(out_dir),"%s/%s",opts->out,r->stages[i].id);
                if(dir_exists(out_dir)){
                    if(!opts->quiet)
                        fprintf(stderr,"  [skip] %s (output exists)\n",r->stages[i].id);
                    r->stages[i].status=2; continue;
                }
            }
            /* Check deps all succeeded */
            int deps_ok=1;
            for(int d=0;d<r->stages[i].n_deps;d++){
                int di=stage_idx(r,r->stages[i].deps[d]);
                if(di>=0&&r->stages[di].status!=2){ deps_ok=0; break; }
            }
            if(!deps_ok){ r->stages[i].status=3; any_failed=1; break; }
            idxs[n_at_level++]=i;
        }
        if(any_failed) break;
        if(n_at_level==0) continue;

        /* Cap parallelism */
        int cap=opts->max_parallel>0?opts->max_parallel:8;
        int batch_start=0;

        while(batch_start<n_at_level){
            int batch_end=batch_start+cap;
            if(batch_end>n_at_level) batch_end=n_at_level;
            int batch_size=batch_end-batch_start;

            /* Spawn all in this batch */
            pid_t pids[MAX_STAGES];
            double t0s[MAX_STAGES];
            for(int b=0;b<batch_size;b++){
                int si=idxs[batch_start+b];
                Stage *s=&r->stages[si];
                char resolved[MAX_ARGS][MAX_ARG_LEN];
                int n_args=0;
                build_argv(s,opts,resolved,&n_args);
                char *argv[MAX_ARGS+1];
                for(int a=0;a<n_args;a++) argv[a]=resolved[a];
                argv[n_args]=NULL;
                if(!opts->quiet)
                    fprintf(stderr,"  [run] (level %d) %s\n",lvl,s->bin);
                t0s[b]=mono_ms();
                pid_t pid=fork();
                if(pid<0){
                    fprintf(stderr,"  [run] fork failed: %s\n",strerror(errno));
                    pids[b]=-1; continue;
                }
                if(pid==0){
                    if(!opts->verbose&&!opts->quiet){
                        int dn=open("/dev/null",1);
                        if(dn>=0){dup2(dn,1);close(dn);}
                    }
                    execvp(s->bin,argv);
                    fprintf(stderr,"  [run] exec failed '%s': %s\n",s->bin,strerror(errno));
                    _exit(127);
                }
                pids[b]=pid;
                s->status=1; /* running */
            }

            /* Wait for all in batch */
            for(int b=0;b<batch_size;b++){
                int si=idxs[batch_start+b];
                Stage *s=&r->stages[si];
                if(pids[b]<0){s->status=3;any_failed=1;continue;}
                int wstatus=0; waitpid(pids[b],&wstatus,0);
                s->wall_ms=mono_ms()-t0s[b];
                s->exit_code=WIFEXITED(wstatus)?WEXITSTATUS(wstatus):-1;
                if(s->exit_code==0){
                    s->status=2;
                    if(!opts->quiet)
                        fprintf(stderr,"  [ok]  %s (%.0f ms)\n",s->id,s->wall_ms);
                } else {
                    s->status=3; any_failed=1;
                    fprintf(stderr,"  [ERR] %s exit=%d\n",s->id,s->exit_code);
                }
            }
            batch_start=batch_end;
        }
    }
    return any_failed?1:0;
}

/* ====================================================================
 * Run manifest
 * ==================================================================== */

static void write_manifest(const Recipe *r, const RunOpts *opts,
                            const char *started_at, double total_wall_ms, int status)
{
    char path[PATH_MAX];
    snprintf(path,sizeof(path),"%s/run-manifest.json",opts->out);
    FILE *fp=fopen(path,"w");
    if(!fp){ fprintf(stderr,"run: cannot write manifest %s\n",path); return; }

    char finished_at[32]; iso_now(finished_at,sizeof(finished_at));
    int ok=0,fail=0,skipped=0;
    for(int i=0;i<r->n_stages;i++){
        if(r->stages[i].status==2) ok++;
        else if(r->stages[i].status==3) fail++;
        else skipped++;
    }

    fprintf(fp,"{\n");
    fprintf(fp,"  \"recipe_code\": \"%s\",\n",r->code);
    fprintf(fp,"  \"recipe_name\": \"%s\",\n",r->name);
    fprintf(fp,"  \"recipe_version\": \"%s\",\n",r->version);
    fprintf(fp,"  \"recipe_hash\": \"%s\",\n",r->hash);
    fprintf(fp,"  \"input\": \"%s\",\n",opts->input);
    fprintf(fp,"  \"out\": \"%s\",\n",opts->out);
    fprintf(fp,"  \"tier\": \"%s\",\n",opts->tier[0]?opts->tier:"free");
    fprintf(fp,"  \"started_at\": \"%s\",\n",started_at);
    fprintf(fp,"  \"finished_at\": \"%s\",\n",finished_at);
    fprintf(fp,"  \"wall_ms\": %.1f,\n",total_wall_ms);
    fprintf(fp,"  \"stages_ok\": %d,\n",ok);
    fprintf(fp,"  \"stages_failed\": %d,\n",fail);
    fprintf(fp,"  \"stages_skipped\": %d,\n",skipped);
    fprintf(fp,"  \"status\": \"%s\",\n",status==0?"ok":"failed");
    fprintf(fp,"  \"stages\": [\n");
    for(int i=0;i<r->n_stages;i++){
        const Stage *s=&r->stages[i];
        const char *st = s->status==2?"ok": s->status==3?"failed":
                         s->status==4?"skipped":"pending";
        fprintf(fp,"    {\"id\":\"%s\",\"bin\":\"%s\","
                   "\"level\":%d,\"status\":\"%s\","
                   "\"exit_code\":%d,\"wall_ms\":%.1f}%s\n",
                s->id,s->bin,s->level,st,s->exit_code,s->wall_ms,
                i<r->n_stages-1?",":"");
    }
    fprintf(fp,"  ]\n}\n");
    fclose(fp);
    if(!opts->quiet) fprintf(stderr,"  [manifest] %s\n",path);
    if(opts->quiet)  printf("%s\n",path);
}

/* ====================================================================
 * Commands
 * ==================================================================== */

static void print_usage(void){
    printf("bonfyre-run %s\n\n",VERSION);
    printf("Usage:\n");
    printf("  bonfyre-run help\n");
    printf("  bonfyre-run show <code>\n");
    printf("  bonfyre-run validate <code>\n");
    printf("  bonfyre-run dry-run <code> <input>\n");
    printf("  bonfyre-run <code> <input> [options]\n\n");
    printf("Options:\n");
    printf("  --out DIR          output directory\n");
    printf("  --dry-run          print plan, do not execute\n");
    printf("  --from-stage ID    start from this stage\n");
    printf("  --to-stage ID      stop after this stage\n");
    printf("  --resume           skip stages whose out dir already exists\n");
    printf("  --quiet            print only final manifest path\n");
    printf("  --verbose          print full stage stdout\n");
    printf("  --tier TIER        pass --tier to all stages\n");
    printf("  --stage-opts S     per-stage flags: \"id:--flag val,...\"\n");
    printf("  --batch DIR        run on every file in DIR\n");
    printf("  --parallel N       max concurrent stages per level (default 8)\n");
    printf("  --db FILE          recipe DB path\n\n");
    printf("Run 'bonfyre-recipe list' to see available codes.\n");
}

static int cmd_show(const char *code, const char *db_path){
    /* Load and pretty-print the stage chain */
    char json[MAX_JSON]={0};
    if(load_recipe_json(code,db_path,json,sizeof(json))!=0){
        fprintf(stderr,"run: recipe '%s' not found\n",code); return 1;
    }
    char name[128]={0},ver[32]={0},desc[512]={0};
    js_str(json,"name",name,sizeof(name));
    js_str(json,"version",ver,sizeof(ver));
    js_str(json,"description",desc,sizeof(desc));
    printf("%s  %s  v%s\n",code,name,ver);
    if(desc[0]) printf("%s\n",desc);
    printf("\nStage chain:\n");

    /* Parse and print stages with level */
    Stage stages[MAX_STAGES]; int n=parse_stages(json,stages,MAX_STAGES);
    Recipe tmp; memset(&tmp,0,sizeof(tmp)); tmp.n_stages=n;
    memcpy(tmp.stages,stages,n*sizeof(Stage));
    assign_levels(&tmp);

    printf("  %-4s  %-16s  %-28s  deps\n","lvl","id","bin");
    printf("  %-4s  %-16s  %-28s  ----\n","---","--","---");
    for(int i=0;i<n;i++){
        Stage *s=&tmp.stages[i];
        char deps[128]={0};
        for(int d=0;d<s->n_deps;d++){
            if(d) strncat(deps,",",sizeof(deps)-strlen(deps)-1);
            strncat(deps,s->deps[d],sizeof(deps)-strlen(deps)-1);
        }
        printf("  %-4d  %-16s  %-28s  %s\n",s->level,s->id,s->bin,deps[0]?deps:"-");
    }
    return 0;
}

static int cmd_validate(const char *code, const char *db_path){
    char json[MAX_JSON]={0};
    if(load_recipe_json(code,db_path,json,sizeof(json))!=0){
        fprintf(stderr,"run: recipe '%s' not found\n",code); return 1;
    }
    Stage stages[MAX_STAGES]; int n=parse_stages(json,stages,MAX_STAGES);
    int ok=0,fail=0;
    for(int i=0;i<n;i++){
        if(stages[i].bin[0]){
            if(on_path(stages[i].bin)){ printf("  OK  %s\n",stages[i].bin); ok++; }
            else { printf("  !!  %s  NOT ON PATH\n",stages[i].bin); fail++; }
        }
    }
    printf("%d/%d binaries found\n",ok,ok+fail);
    return fail>0?1:0;
}

static int cmd_dry_run(const char *code, const RunOpts *opts){
    char json[MAX_JSON]={0};
    if(load_recipe_json(code,opts->db_path,json,sizeof(json))!=0){
        fprintf(stderr,"run: recipe '%s' not found\n",code); return 1;
    }
    char name[128]={0},ver[32]={0};
    js_str(json,"name",name,sizeof(name));
    js_str(json,"version",ver,sizeof(ver));
    printf("=== DRY RUN: %s  (%s, v%s) ===\n",code,name,ver);
    printf("input: %s\nout:   %s\ntier:  %s\n\n",
           opts->input,opts->out,opts->tier[0]?opts->tier:"free");

    Stage stages[MAX_STAGES]; int n=parse_stages(json,stages,MAX_STAGES);
    Recipe tmp; memset(&tmp,0,sizeof(tmp)); tmp.n_stages=n;
    memcpy(tmp.stages,stages,n*sizeof(Stage)); assign_levels(&tmp);

    printf("Execution plan (%d stages):\n",n);
    int max_lvl=0;
    for(int i=0;i<n;i++) if(tmp.stages[i].level>max_lvl) max_lvl=tmp.stages[i].level;

    for(int lvl=0;lvl<=max_lvl;lvl++){
        printf("\n  [Level %d]%s\n",lvl,lvl>0?" (waits for level above)":"");
        for(int i=0;i<n;i++){
            if(tmp.stages[i].level!=lvl) continue;
            Stage *s=&tmp.stages[i];
            printf("    %s\n      cmd: %s",s->id,s->bin);
            for(int a=0;a<s->n_args;a++){
                char resolved[MAX_ARG_LEN];
                subst(s->args[a],opts->input,opts->out,resolved,sizeof(resolved));
                printf(" %s",resolved);
            }
            if(opts->tier[0]) printf(" --tier %s",opts->tier);
            printf("\n");
        }
    }
    printf("\nOutput dir: %s\nManifest: %s/run-manifest.json\n",opts->out,opts->out);
    return 0;
}

static int run_single(const char *code, RunOpts *opts){
    Recipe r; memset(&r,0,sizeof(r));
    if(load_recipe(code,opts->db_path,&r)!=0) return 1;
    assign_levels(&r);

    if(opts->dry_run) return cmd_dry_run(code,opts);

    /* Validate all binaries before starting */
    int missing=0;
    for(int i=0;i<r.n_stages;i++)
        if(r.stages[i].bin[0]&&!on_path(r.stages[i].bin)){
            fprintf(stderr,"run: binary not found: %s (stage: %s)\n",
                    r.stages[i].bin,r.stages[i].id);
            missing++;
        }
    if(missing){
        fprintf(stderr,"run: %d missing binaries — aborting. "
                       "Run 'bonfyre-run validate %s' for full report.\n",missing,code);
        return 1;
    }

    /* Create output directory */
    struct stat st;
    if(stat(opts->out,&st)!=0){
        /* mkdir -p */
        char tmp[PATH_MAX]; strncpy(tmp,opts->out,sizeof(tmp)-1);
        for(char *p=tmp+1;*p;p++){
            if(*p=='/'){ *p='\0'; mkdir(tmp,0755); *p='/'; }
        }
        if(mkdir(opts->out,0755)!=0&&errno!=EEXIST){
            fprintf(stderr,"run: cannot create output dir %s: %s\n",opts->out,strerror(errno));
            return 1;
        }
    }

    char started_at[32]; iso_now(started_at,sizeof(started_at));
    if(!opts->quiet)
        fprintf(stderr,"[bonfyre-run] %s  input=%s  out=%s\n",code,opts->input,opts->out);

    double t0=mono_ms();
    int rc=stages_run(&r,opts);
    double total_ms=mono_ms()-t0;

    write_manifest(&r,opts,started_at,total_ms,rc);

    if(!opts->quiet){
        int ok=0;
        for(int i=0;i<r.n_stages;i++){
            if(r.stages[i].status==2) ok++;
        }
        fprintf(stderr,"[bonfyre-run] %s  %d/%d stages ok  %.0f ms\n",
                code,ok,r.n_stages,total_ms);
    }
    return rc;
}

static int run_batch(const char *code, RunOpts *base_opts){
    DIR *d=opendir(base_opts->batch_dir);
    if(!d){ fprintf(stderr,"run: cannot open batch dir %s\n",base_opts->batch_dir); return 1; }
    int ok=0,fail=0;
    struct dirent *ent;
    while((ent=readdir(d))!=NULL){
        if(ent->d_name[0]=='.') continue;
        char filepath[PATH_MAX];
        snprintf(filepath,sizeof(filepath),"%s/%s",base_opts->batch_dir,ent->d_name);
        struct stat st; if(stat(filepath,&st)!=0||S_ISDIR(st.st_mode)) continue;

        RunOpts opts=*base_opts;
        strncpy(opts.input,filepath,sizeof(opts.input)-1);
        /* Output dir: base_opts->out/<filename-without-ext> */
        char stem[256]; strncpy(stem,ent->d_name,sizeof(stem)-1);
        char *dot=strrchr(stem,'.'); if(dot) *dot='\0';
        snprintf(opts.out,sizeof(opts.out),"%s/%s",base_opts->out,stem);

        fprintf(stderr,"[batch] %s\n",filepath);
        if(run_single(code,&opts)==0) ok++;
        else fail++;
    }
    closedir(d);
    fprintf(stderr,"[batch] done: %d ok, %d failed\n",ok,fail);
    return fail>0?1:0;
}

/* ====================================================================
 * Argument parsing + main
 * ==================================================================== */

int main(int argc, char **argv){
    if(argc<2||strcmp(argv[1],"help")==0||strcmp(argv[1],"--help")==0){
        print_usage(); return 0;
    }

    if(strcmp(argv[1],"show")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-run show <code>\n");return 1;}
        char db[PATH_MAX]; strncpy(db,default_db_path(),sizeof(db)-1);
        return cmd_show(argv[2],db);
    }
    if(strcmp(argv[1],"validate")==0){
        if(argc<3){fprintf(stderr,"usage: bonfyre-run validate <code>\n");return 1;}
        char db[PATH_MAX]; strncpy(db,default_db_path(),sizeof(db)-1);
        return cmd_validate(argv[2],db);
    }
    if(strcmp(argv[1],"dry-run")==0){
        if(argc<4){fprintf(stderr,"usage: bonfyre-run dry-run <code> <input>\n");return 1;}
        RunOpts opts; memset(&opts,0,sizeof(opts));
        opts.max_parallel=8; opts.dry_run=1;
        strncpy(opts.db_path,default_db_path(),sizeof(opts.db_path)-1);
        strncpy(opts.input,argv[3],sizeof(opts.input)-1);
        snprintf(opts.out,sizeof(opts.out),"./bonfyre-out/dry-run");
        return cmd_dry_run(argv[2],&opts);
    }

    /* Default mode: bonfyre-run <code> <input> [opts] */
    if(argc<3){print_usage();return 1;}
    const char *code=argv[1];

    RunOpts opts; memset(&opts,0,sizeof(opts));
    opts.max_parallel=8;
    strncpy(opts.db_path,default_db_path(),sizeof(opts.db_path)-1);
    strncpy(opts.input,argv[2],sizeof(opts.input)-1);

    /* Default out dir: ./bonfyre-out/<timestamp> */
    {
        char ts[32]; time_t t=time(NULL); struct tm tm; gmtime_r(&t,&tm);
        strftime(ts,sizeof(ts),"%Y%m%dT%H%M%SZ",&tm);
        snprintf(opts.out,sizeof(opts.out),"./bonfyre-out/%s-%s",code,ts);
    }

    /* Parse options */
    for(int i=3;i<argc;i++){
        if(strcmp(argv[i],"--dry-run")==0)               opts.dry_run=1;
        else if(strcmp(argv[i],"--resume")==0)            opts.resume=1;
        else if(strcmp(argv[i],"--quiet")==0)             opts.quiet=1;
        else if(strcmp(argv[i],"--verbose")==0)           opts.verbose=1;
        else if(strcmp(argv[i],"--out")==0&&i+1<argc)    {strncpy(opts.out,argv[++i],sizeof(opts.out)-1);}
        else if(strcmp(argv[i],"--from-stage")==0&&i+1<argc) {strncpy(opts.from_stage,argv[++i],sizeof(opts.from_stage)-1);}
        else if(strcmp(argv[i],"--to-stage")==0&&i+1<argc)   {strncpy(opts.to_stage,  argv[++i],sizeof(opts.to_stage)-1);}
        else if(strcmp(argv[i],"--tier")==0&&i+1<argc)       {strncpy(opts.tier,       argv[++i],sizeof(opts.tier)-1);}
        else if(strcmp(argv[i],"--stage-opts")==0&&i+1<argc) {strncpy(opts.stage_opts, argv[++i],sizeof(opts.stage_opts)-1);}
        else if(strcmp(argv[i],"--parallel")==0&&i+1<argc)   {opts.max_parallel=atoi(argv[++i]);}
        else if(strcmp(argv[i],"--db")==0&&i+1<argc)         {strncpy(opts.db_path,    argv[++i],sizeof(opts.db_path)-1);}
        else if(strcmp(argv[i],"--batch")==0&&i+1<argc){
            opts.batch=1; strncpy(opts.batch_dir,argv[++i],sizeof(opts.batch_dir)-1);
        }
        else if(argv[i][0]=='-'){
            fprintf(stderr,"run: unknown option '%s'\n",argv[i]); return 1;
        }
    }

    if(opts.dry_run) return cmd_dry_run(code,&opts);
    if(opts.batch)   return run_batch(code,&opts);
    return run_single(code,&opts);
}
