// Copyright (c) 2018-2020 Osamu Hirose
// Modified for Python CFFI wrapping

#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<string.h>
#include<math.h>
#include<unistd.h>
#include<ctype.h>
#include<time.h>
#include<sys/time.h>
#include"../base/util.h"
#include"../base/misc.h"
#include"../base/kdtree.h"
#include"../base/kernel.h"
#include"../base/sampling.h"
#include"../base/sgraph.h"
#include"../base/geokdecomp.h"
#include"bcpd.h"
#include"info.h"
#include"norm.h"

#define SQ(x) ((x)*(x))
#define SWAP(x, y) { int temp = x; x = y; y = temp; }

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 声明一些外部函数
void init_genrand64(unsigned long s);
enum transpose {ASIS=0,TRANSPOSE=1};

// 参数类型枚举，对应于getopt中的字符选项
typedef enum {
    OPT_U, OPT_T, OPT_t, OPT_X, OPT_Y, OPT_C, OPT_D, OPT_z, OPT_u, OPT_r,
    OPT_w, OPT_l, OPT_b, OPT_k, OPT_g, OPT_d, OPT_e, OPT_c, OPT_n, OPT_N,
    OPT_G, OPT_J, OPT_K, OPT_o, OPT_x, OPT_y, OPT_f, OPT_s, OPT_h, OPT_i,
    OPT_p, OPT_q, OPT_v, OPT_a, OPT_A, OPT_W
} opt_type;

// 参数结构体，用于存储选项和参数值
typedef struct {
    opt_type type;
    char* value;  // 对于带参数的选项
} cmd_option;

// 模拟getopt的函数，处理一个命令行选项
void process_opt(pwpm *pm, opt_type type, char* optarg) {
    switch(type) {
        case OPT_D: scan_dwpm(pm->dwn, pm->dwr, optarg);  break;
        case OPT_G: scan_kernel(pm, optarg);              break;
        case OPT_t: pm->fpv  = atof(optarg);              break;
        case OPT_z: pm->eps  = atof(optarg);              break;
        case OPT_b: pm->bet  = atof(optarg);              break;
        case OPT_w: pm->omg  = atof(optarg);              break;
        case OPT_l: pm->lmd  = atof(optarg);              break;
        case OPT_k: pm->kpa  = atof(optarg);              break;
        case OPT_g: pm->gma  = atof(optarg);              break;
        case OPT_d: pm->dlt  = atof(optarg);              break;
        case OPT_e: pm->lim  = atof(optarg);              break;
        case OPT_f: pm->btn  = atof(optarg);              break;
        case OPT_c: pm->cnv  = atof(optarg);              break;
        case OPT_n: pm->nlp  = atoi(optarg);              break;
        case OPT_N: pm->llp  = atoi(optarg);              break;
        case OPT_K: pm->K    = atoi(optarg);              break;
        case OPT_J: pm->J    = atoi(optarg);              break;
        case OPT_r: pm->rns  = atoi(optarg);              break;
        case OPT_u: pm->nrm  = *optarg;                   break;
        case OPT_U: pm->nrf  = *optarg;                   break;
        case OPT_h: pm->opt |= PW_OPT_HISTO;              break;
        case OPT_a: pm->opt |= PW_OPT_DBIAS;              break;
        case OPT_p: pm->opt |= PW_OPT_LOCAL;              break;
        case OPT_q: pm->opt |= PW_OPT_QUIET;              break;
        case OPT_A: pm->opt |= PW_OPT_ACCEL;              break;
        case OPT_W: pm->opt |= PW_OPT_NWARN;              break;
        case OPT_i: pm->opt |= PW_OPT_ISOVF;              break;
        case OPT_o: strcpy(pm->fn[OUTPUT], optarg);       break;
        case OPT_x: strcpy(pm->fn[TARGET], optarg);       break;
        case OPT_y: strcpy(pm->fn[SOURCE], optarg);       break;
        case OPT_X: strcpy(pm->fn[FUNC_X], optarg);       break;
        case OPT_Y: strcpy(pm->fn[FUNC_Y], optarg);       break;
        case OPT_C: strcpy(pm->fn[COV_LQ], optarg);       break;
        case OPT_v: /* 忽略帮助 */                         break;
        case OPT_T:
            if(1==strlen(optarg)) assert(!strchr(optarg,'s'));
            if( strchr(optarg,'a')) pm->opt |= PW_OPT_AFFIN;
            if(!strchr(optarg,'s')) pm->opt |= PW_OPT_NOSCL;
            if(!strchr(optarg,'n')) pm->opt |= PW_OPT_NONRG;
            if(!strchr(optarg,'r')) pm->opt |= PW_OPT_NOSIM;
            break;
        case OPT_s:
            if(strchr(optarg,'A')) pm->opt |= PW_OPT_SAVE;
            if(strchr(optarg,'x')) pm->opt |= PW_OPT_SAVEX;
            if(strchr(optarg,'y')) pm->opt |= PW_OPT_SAVEY;
            if(strchr(optarg,'v')) pm->opt |= PW_OPT_SAVEV;
            if(strchr(optarg,'a')) pm->opt |= PW_OPT_SAVEA;
            if(strchr(optarg,'c')) pm->opt |= PW_OPT_SAVEC;
            if(strchr(optarg,'e')) pm->opt |= PW_OPT_SAVEE;
            if(strchr(optarg,'P')) pm->opt |= PW_OPT_SAVEP;
            if(strchr(optarg,'T')) pm->opt |= PW_OPT_SAVET;
            if(strchr(optarg,'X')) pm->opt |= PW_OPT_PATHX;
            if(strchr(optarg,'Y')) pm->opt |= PW_OPT_PATHY;
            if(strchr(optarg,'t')) pm->opt |= PW_OPT_PFLOG;
            if(strchr(optarg,'0')) pm->opt |= PW_OPT_VTIME;
            break;
    }
}

// 根据命令行对应字符转换为选项枚举
opt_type char_to_opt_type(char c) {
    switch(c) {
        case 'U': return OPT_U;
        case 'T': return OPT_T;
        case 't': return OPT_t;
        case 'X': return OPT_X;
        case 'Y': return OPT_Y;
        case 'C': return OPT_C;
        case 'D': return OPT_D;
        case 'z': return OPT_z;
        case 'u': return OPT_u;
        case 'r': return OPT_r;
        case 'w': return OPT_w;
        case 'l': return OPT_l;
        case 'b': return OPT_b;
        case 'k': return OPT_k;
        case 'g': return OPT_g;
        case 'd': return OPT_d;
        case 'e': return OPT_e;
        case 'c': return OPT_c;
        case 'n': return OPT_n;
        case 'N': return OPT_N;
        case 'G': return OPT_G;
        case 'J': return OPT_J;
        case 'K': return OPT_K;
        case 'o': return OPT_o;
        case 'x': return OPT_x;
        case 'y': return OPT_y;
        case 'f': return OPT_f;
        case 's': return OPT_s;
        case 'h': return OPT_h;
        case 'i': return OPT_i;
        case 'p': return OPT_p;
        case 'q': return OPT_q;
        case 'v': return OPT_v;
        case 'a': return OPT_a;
        case 'A': return OPT_A;
        case 'W': return OPT_W;
        default:  return -1;  // 未知选项
    }
}

// opt_type -> 字符的映射函数
char opt_type_to_char(opt_type type) {
    switch(type) {
        case OPT_U: return 'U';
        case OPT_T: return 'T';
        case OPT_t: return 't';
        case OPT_X: return 'X';
        case OPT_Y: return 'Y';
        case OPT_C: return 'C';
        case OPT_D: return 'D';
        case OPT_z: return 'z';
        case OPT_u: return 'u';
        case OPT_r: return 'r';
        case OPT_w: return 'w';
        case OPT_l: return 'l';
        case OPT_b: return 'b';
        case OPT_k: return 'k';
        case OPT_g: return 'g';
        case OPT_d: return 'd';
        case OPT_e: return 'e';
        case OPT_c: return 'c';
        case OPT_n: return 'n';
        case OPT_N: return 'N';
        case OPT_G: return 'G';
        case OPT_J: return 'J';
        case OPT_K: return 'K';
        case OPT_o: return 'o';
        case OPT_x: return 'x';
        case OPT_y: return 'y';
        case OPT_f: return 'f';
        case OPT_s: return 's';
        case OPT_h: return 'h';
        case OPT_i: return 'i';
        case OPT_p: return 'p';
        case OPT_q: return 'q';
        case OPT_v: return 'v';
        case OPT_a: return 'a';
        case OPT_A: return 'A';
        case OPT_W: return 'W';
        default:    return '?'; // 未知
    }
}
// 修改后的pw_getopt函数，通过选项数组直接处理参数
void pw_getopt_direct(pwpm *pm, cmd_option* options, int num_options) {
    int i;
    
    // 设置默认值
    strcpy(pm->fn[TARGET],"X.txt");   pm->omg=0.0; pm->cnv=1e-4; pm->K=0; pm->opt=0.0; pm->btn=0.20; pm->bet=2.0;
    strcpy(pm->fn[SOURCE],"Y.txt");   pm->lmd=2.0; pm->nlp= 500; pm->J=0; pm->dlt=7.0; pm->lim=0.15; pm->eps=1e-3;
    strcpy(pm->fn[OUTPUT],"output_"); pm->rns=0;   pm->llp=  30; pm->G=0; pm->gma=1.0; pm->kpa=ZERO; pm->nrm='e';
    strcpy(pm->fn[FACE_Y],"");        pm->nnk=0;   pm->nnr=   0;          pm->fpv=0.1; pm->nrf='e';
    strcpy(pm->fn[FUNC_Y],"");
    strcpy(pm->fn[FUNC_X],"");
    strcpy(pm->fn[COV_LQ],"");
    pm->dwn[SOURCE]=0; pm->dwr[SOURCE]=0.0f;
    pm->dwn[TARGET]=0; pm->dwr[TARGET]=0.0f;

    // // 处理每个选项
    // printf("\n==== cmd_option 参数列表 (共 %d 个) ====\n", num_options);
    // for (int i = 0; i < num_options; i++) {
    //     char chr = opt_type_to_char(options[i].type);
    //     printf("  [%02d] type=%2d ('%c') value=%p", i, options[i].type, chr, (void*)options[i].value);
    //     if (options[i].value)
    //         printf("  content=\"%s\"\n", options[i].value);
    //     else
    //         printf("  content=NULL\n");
    // }
    // printf("======================================\n");
    
    // 处理每个选项
    for (i = 0; i < num_options; i++) {
        process_opt(pm, options[i].type, options[i].value);
    }
    
    // 后处理逻辑，与原始函数相同
    if(pm->opt&PW_OPT_AFFIN) pm->opt |=PW_OPT_NOSIM|PW_OPT_NOSCL;
    /* disable 'for each' normalization if rigid */
    if((pm->opt&PW_OPT_NONRG)&&(pm->opt&PW_OPT_NOSCL)) if(pm->nrm=='e') pm->nrm='y';
    /* acceleration with default parameters */
    if(pm->opt&PW_OPT_ACCEL) {pm->J=300;pm->K=70;pm->opt|=PW_OPT_LOCAL;}
    /* case: save all */
    if(pm->opt&PW_OPT_SAVE)
        pm->opt|=PW_OPT_SAVEX|PW_OPT_SAVEC|PW_OPT_SAVEP|PW_OPT_SAVEA|PW_OPT_PFLOG|
                 PW_OPT_SAVEY|PW_OPT_SAVEV|PW_OPT_SAVEE|PW_OPT_SAVET|PW_OPT_PATHY;
    /* always save y & info */
    pm->opt |= PW_OPT_SAVEY|PW_OPT_INFO;
    /* for numerical stability */
    pm->omg=pm->omg==0?1e-250:pm->omg;
    /* llp is always less than or equal to nlp */
    if(pm->llp>pm->nlp) pm->llp=pm->nlp;
    // printf("pm->btn: %f\n", pm->btn);
    return;
}

// 辅助函数：创建一个选项
// cmd_option create_option(char opt_char, const char* value) {
//     cmd_option opt;
//     opt.type = char_to_opt_type(opt_char);
//     if (value) {
//         opt.value = strdup(value);
//     } else {
//         opt.value = NULL;
//     }
//     return opt;
// }

cmd_option create_option(char opt_char, const char* value) {
    cmd_option opt;
    opt.type = char_to_opt_type(opt_char);
    
    if (value) {
        opt.value = (char*)malloc(strlen(value) + 1);
        // opt.value = malloc(strlen(value) + 1);
        // strcpy(opt.value, value);
        if (opt.value == NULL) {
            fprintf(stderr, "内存分配失败在create_option\n");
            exit(EXIT_FAILURE);
        }
        strcpy(opt.value, value);
    } else {
        opt.value = NULL;
    }
    
    return opt;
}

// 辅助函数：释放选项数组
void free_options(cmd_option* options, int num_options) {
    if (!options) return;
    
    for (int i = 0; i < num_options; i++) {
        if (options[i].value) {
            free(options[i].value);
        }
    }
    if (options) free(options);
}

// 将 double 转换为字符串并设置选项
void set_double_option(cmd_option** options, int* index, char option_char, double value) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%.6f", value);
    (*options)[*index] = create_option(option_char, buffer);
    (*index)++;
}

// 在bcpd_wrapper中构建选项数组
void build_bcpd_options(
    cmd_option** options, int* num_options,
    double omega, double lambda, double beta, double gamma, double kappa,
    int max_iter, int min_iter, double tolerance,
    int nystrom_J, int nystrom_K, int use_kdtree, int random_seed,
    int history_mode, int quiet_mode, int accel_mode,
    char normalization_type,
    int gauss_kernel_type, double tau, int knn, double radius,
    int transform_type,
    double search_scale, double search_radius, double sigma_threshold,
    int save_options,
    int dwn_X, int dwn_Y, double dwr_X, double dwr_Y)
{
    char buffer[32]; // 临时字符串缓冲区
    int i = 0;
    
    // 为了安全起见，分配一个足够大的空间
    // 根据您的情况，已知实际需要21个选项，我们分配25个以确保足够
    int count = 25;
    
    // 分配选项数组
    *options = (cmd_option*)malloc(count * sizeof(cmd_option));
    if (*options == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }
    
    // 设置所有选项
    set_double_option(options, &i, 'w', omega);
    set_double_option(options, &i, 'l', lambda);
    set_double_option(options, &i, 'b', beta);
    set_double_option(options, &i, 'g', gamma);
    

    snprintf(buffer, sizeof(buffer), "%d", random_seed);
    (*options)[i++] = create_option('r', buffer);
    
    snprintf(buffer, sizeof(buffer), "%d", max_iter);
    (*options)[i++] = create_option('n', buffer);
    
    snprintf(buffer, sizeof(buffer), "%d", min_iter);
    (*options)[i++] = create_option('N', buffer);
    
    // snprintf(buffer, "%.6e", tolerance);
    // (*options)[i++] = create_option('c', buffer);
    set_double_option(options, &i, 'c', tolerance);
    
    // printf("Building options->nystrom_J= %d\n", nystrom_J);
    snprintf(buffer, sizeof(buffer), "%d", nystrom_J);
    (*options)[i++] = create_option('J', buffer);
    
    snprintf(buffer, sizeof(buffer), "%d", nystrom_K);
    (*options)[i++] = create_option('K', buffer);
    // printf("buffer = '%s'\n", buffer);
    
    // 归一化类型
    snprintf(buffer, sizeof(buffer), "%c", normalization_type);
    (*options)[i++] = create_option('u', buffer);
    
    // 搜索参数
    // snprintf(buffer, "%.6f", search_scale);
    snprintf(buffer, sizeof(buffer), "%.6f", search_scale);
    (*options)[i++] = create_option('d', buffer);
    
    // snprintf(buffer, "%.6f", search_radius);
    snprintf(buffer, sizeof(buffer), "%.6f", search_radius);
    (*options)[i++] = create_option('e', buffer);
    
    // snprintf(buffer, "%.6f", sigma_threshold);
    snprintf(buffer, sizeof(buffer), "%.6f", sigma_threshold);
    (*options)[i++] = create_option('f', buffer);
    
    // 条件选项
    if (history_mode) (*options)[i++] = create_option('h', NULL);
    if (quiet_mode) (*options)[i++] = create_option('q', NULL);
    if (use_kdtree) (*options)[i++] = create_option('p', NULL);
    if (accel_mode) (*options)[i++] = create_option('A', NULL);
    
    // 变换类型
    if (transform_type > 0) {
        switch(transform_type) {
            case 1: (*options)[i++] = create_option('T', "a"); break;  // Tan
            case 2: (*options)[i++] = create_option('T', "a,n"); break; // Ta
            case 3: (*options)[i++] = create_option('T', "n"); break;   // Tsr
            case 4: (*options)[i++] = create_option('T', "s,n"); break; // Tr
            case 5: (*options)[i++] = create_option('T', "s,r"); break; // Tn
        }
    }
    
    // 保存选项
    if (save_options > 0) {
        char save_str[32] = "";
        if (save_options & 0x0001) strcat(save_str, "x");
        if (save_options & 0x0002) strcat(save_str, "y");
        if (save_options & 0x0004) strcat(save_str, "v");
        if (save_options & 0x0008) strcat(save_str, "a");
        if (save_options & 0x0010) strcat(save_str, "c");
        if (save_options & 0x0020) strcat(save_str, "e");
        if (save_options & 0x0040) strcat(save_str, "P");
        if (save_options & 0x0080) strcat(save_str, "T");
        if (save_options & 0x0100) strcat(save_str, "X");
        if (save_options & 0x0200) strcat(save_str, "Y");
        if (save_options & 0x0400) strcat(save_str, "t");
        if (save_options & 0x0800) strcat(save_str, "0");
        if (save_options & 0x1000) strcat(save_str, "A");
        
        if (strlen(save_str) > 0) {
            (*options)[i++] = create_option('s', save_str);
        }
    }
    
    // 下采样选项
    if (dwn_X > 0 || dwn_Y > 0) {
        snprintf(buffer, sizeof(buffer), "%d:%f,%d:%f", dwn_X, dwr_X, dwn_Y, dwr_Y);
        (*options)[i++] = create_option('D', buffer);
    }
    
    // 核函数类型
    if (gauss_kernel_type != 0) {
        if (gauss_kernel_type == -1) {
            snprintf(buffer, sizeof(buffer), "%d:%f:%d:%f", gauss_kernel_type, tau, knn, radius);
        } else {
            snprintf(buffer, sizeof(buffer), "%d", gauss_kernel_type);
        }
        (*options)[i++] = create_option('G', buffer);
    }
    
    // 使用实际设置的选项数量
    *num_options = i;
    
    // // 如果需要，可以重新分配内存以精确匹配实际选项数量
    // // *options = (cmd_option*)realloc(*options, i * sizeof(cmd_option));
    // printf("\n==== 选项内存内容 (共 %d 个) ====\n", *num_options);
    // for (int j = 0; j < *num_options; j++) {
    //     printf("选项 %2d: 类型值=%d, 值指针=%p\n", 
    //         j, (*options)[j].type, (void*)(*options)[j].value);
        
    //     if ((*options)[j].value != NULL) {
    //         printf("  值内容: \"%s\"\n", (*options)[j].value);
    //     }
    // }
    // printf("=========================\n\n");
    
    // 打印实际使用的选项数量以便调试
    // fprintf(stderr, "实际设置的选项数量: %d\n", i);
    return;
}
  
// 保留main.c原有的所有辅助函数
void save_variable(const char *prefix, const char *suffix,const double *var, int D, int J, char *fmt, int trans){
  int d,j; char fn[256]; double **buf;
  strcpy(fn,prefix); strcat(fn,suffix);
  if(trans==TRANSPOSE){
    buf=calloc2d(J,D);
    for(j=0;j<J;j++)for(d=0;d<D;d++) buf[j][d]=var[d+D*j];
    write2d(fn,(const double **)buf,J,D,fmt,"NA"); free2d(buf,J);
  }
  else {
    buf=calloc2d(D,J);
    for(j=0;j<J;j++)for(d=0;d<D;d++) buf[d][j]=var[d+D*j];
    write2d(fn,(const double **)buf,D,J,fmt,"NA"); free2d(buf,D);
  }

  return;
}

void save_corresp(
    const char   *prefix,
    const double *X,
    const double *y,
    const double *a,
    const double *sgm,
    const double  s,
    const double  r,
    pwsz          sz,
    pwpm          pm
  ){
  int i,m,n,D,M,N; int *T,*l,*bi; double *bd; double *p,c,val; char fnP[256],fnc[256],fne[256];
  int S[MAXTREEDEPTH]; int top,ct; double omg,dlt,vol,rad; int si=sizeof(int),sd=sizeof(double);
  FILE *fpP=NULL,*fpc=NULL,*fpe=NULL; int db=pm.opt&PW_OPT_DBIAS; double max,min; int mmax;

  D=sz.D; M=sz.M; N=sz.N; omg=pm.omg; dlt=pm.dlt; rad=dlt*r;
  strcpy(fnP,prefix);strcat(fnP,"P.txt");if(pm.opt&PW_OPT_SAVEP){fpP=fopen(fnP,"w");fprintf(fpP,"[n]\t[m]\t[probability]\n");}
  strcpy(fne,prefix);strcat(fne,"e.txt");if(pm.opt&PW_OPT_SAVEE){fpe=fopen(fne,"w");fprintf(fpe,"[n]\t[m]\t[probability]\n");}
  strcpy(fnc,prefix);strcat(fnc,"c.txt");if(pm.opt&PW_OPT_SAVEC){fpc=fopen(fnc,"w");fprintf(fpc,"[n]\t[1/0]\n");}

  T=calloc(3*M+1,si); bi=calloc(6*M,si); bd=calloc(2*M,sd); p=calloc(M,sd); l=calloc(M,si); 
  kdtree(T,bi,bd,y,D,M); vol=volume(X,D,N); c=(pow(2.0*M_PI*SQ(r),0.5*D)*omg)/(vol*(1-omg));
  for(n=0;n<N;n++){
    /* compute P, c, e */
    val=c;top=ct=0;do{eballsearch_next(&m,S,&top,X+D*n,rad,y,T,D,M);if(m>=0)l[ct++]=m;}while(top);
    if(!ct){nnsearch(&m,&min,X+D*n,y,T,D,M);l[ct++]=m;}
    for(i=0;i<ct;i++){m=l[i];p[i]=a[m]*gauss(y+D*m,X+D*n,D,r)*(db?exp(-0.5*D*SQ(sgm[m]*s/r)):1.0);val+=p[i];}
    for(i=0;i<ct;i++){m=l[i];p[i]/=val;}
    max=c/val;mmax=0;for(i=0;i<ct;i++)if(p[i]>max){max=p[i];mmax=l[i]+1;}
    /* print P, c, e */
    if(fpP){for(i=0;i<ct;i++)if(p[i]>1.0f/M){m=l[i];fprintf(fpP,"%d\t%d\t%lf\n",n+1,m+1,p[i]);}}
    if(fpe){fprintf(fpe,"%d\t%d\t%lf\n",n+1,mmax?mmax:l[0],mmax?max:p[0]);}
    if(fpc){fprintf(fpc,"%d\t%d\n",n+1,mmax?1:0);}
  }
  if(fpP){fclose(fpP);} free(l); free(bd);
  if(fpe){fclose(fpc);} free(p); free(bi);
  if(fpc){fclose(fpe);} free(T);
  return;
}

int save_optpath(const char *file, const double *sy, const double *X, pwsz sz, pwpm pm, int lp){
  int N=sz.N,M=sz.M,D=sz.D; int si=sizeof(int),sd=sizeof(double);
  FILE *fp=fopen(file,"wb");
  if(!fp){printf("Can't open: %s\n",file);exit(EXIT_FAILURE);}
  fwrite(&N, si,   1,  fp);
  fwrite(&D, si,   1,  fp);
  fwrite(&M, si,   1,  fp);
  fwrite(&lp,si,   1,  fp);
  fwrite(sy, sd,lp*D*M,fp);
  fwrite(X,  sd,   D*N,fp);
  if(strlen(pm.fn[FACE_Y])){double *b; int nl,nc,l,c; int *L=NULL;
    b=read2dcm(&nl,&nc,pm.fn[FACE_Y]); assert(nc==3||nc==2);
    L=calloc(nc*nl,si); for(l=0;l<nl;l++)for(c=0;c<nc;c++){L[c+nc*l]=(int)b[c+nc*l];}
    fwrite(&nl,si,  1,  fp);
    fwrite(&nc,si,  1,  fp);
    fwrite(L,  si,nc*nl,fp);
    free(L); free(b);
  }
  fclose(fp);

  return 0;
}

void scan_kernel(pwpm *pm, const char *arg){ char *p;
  if('g'!=tolower(*arg)) {pm->G=atoi(arg);return;}
  p=strchr(arg,','); if(!p){printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");exit(EXIT_FAILURE);}
  if(strstr(arg,"txt")) sscanf(p+1,"%lf,%s",    &(pm->tau),pm->fn[FACE_Y]);
  else                  sscanf(p+1,"%lf,%d,%lf",&(pm->tau),&(pm->nnk),&(pm->nnr));
  if(pm->tau<0||pm->tau>1){printf("ERROR: the 2nd argument of -G (tau) must be in range [0,1]. Abort.\n"); exit(EXIT_FAILURE);}
}

void scan_dwpm(int *dwn, double *dwr, const char *arg){
  char c; int n,m; double r;
  m=sscanf(arg,"%c,%d,%lf",&c,&n,&r);
  if(m!=3) goto err01;
  if(n<=0) goto err03;
  if(r< 0) goto err04;
  if(isupper(c)){r*=-1.0f;} c=tolower(c);
  if(c!='x'&&c!='y'&&c!='b') goto err02;
  if(r<0&&-r<1e-2) r=-1e-2;
  switch(c){
    case 'x': dwn[TARGET]=n; dwr[TARGET]=r; break;
    case 'y': dwn[SOURCE]=n; dwr[SOURCE]=r; break;
    case 'b': dwn[TARGET]=n; dwr[TARGET]=r;
              dwn[SOURCE]=n; dwr[SOURCE]=r; break;
  }
  return;
  err01: printf("ERROR: The argument of '-D' must be 'char,int,real'. \n");         exit(EXIT_FAILURE);
  err02: printf("ERROR: The 1st argument of '-D' must be one of [x,y,b,X,Y,B]. \n");exit(EXIT_FAILURE);
  err03: printf("ERROR: The 2nd argument of '-D' must be positive.     \n");        exit(EXIT_FAILURE);
  err04: printf("ERROR: The 3rd argument of '-D' must be positive or 0.\n");        exit(EXIT_FAILURE);
}

void check_prms(const pwpm pm, const pwsz sz){
  int M=sz.M,N=sz.N,M0=pm.dwn[SOURCE],N0=pm.dwn[TARGET]; M=M0?M0:M; N=N0?N0:N;
  if(pm.nlp<=0){printf("ERROR: -n: Argument must be a positive integer. Abort.\n"); exit(EXIT_FAILURE);}
  if(pm.llp<=0){printf("ERROR: -N: Argument must be a positive integer. Abort.\n"); exit(EXIT_FAILURE);}
  if(pm.omg< 0){printf("ERROR: -w: Argument must be in range [0,1]. Abort.\n");     exit(EXIT_FAILURE);}
  if(pm.omg>=1){printf("ERROR: -w: Argument must be in range [0,1]. Abort.\n");     exit(EXIT_FAILURE);}
  if(pm.lmd<=0){printf("ERROR: -l: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.kpa<=0){printf("ERROR: -k: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.dlt<=0){printf("ERROR: -d: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.lim<=0){printf("ERROR: -e: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.btn<=0){printf("ERROR: -f: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.cnv<=0){printf("ERROR: -c: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.rns< 0){printf("ERROR: -r: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.K<0)   {printf("ERROR: -K: Argument must be a positive integer. Abort.\n"); exit(EXIT_FAILURE);}
  if(pm.K>M)   {printf("ERROR: -K: Argument must be less than M. Abort.\n");        exit(EXIT_FAILURE);}
  if(pm.J<0)   {printf("ERROR: -J: Argument must be a positive integer. Abort.\n"); exit(EXIT_FAILURE);}
//   printf("pm.J=%d, M=%d, N=%d, M+N=%d\n", pm.J, M, N, M+N);
  if(pm.J>(M+N)) {printf("ERROR: -J: Argument must be less than M+N. Abort.\n");      exit(EXIT_FAILURE);}
  if(pm.G>3)   {printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");     exit(EXIT_FAILURE);}
  if(pm.G<0)   {printf("ERROR: -G: Arguments are wrongly specified. Abort.\n");     exit(EXIT_FAILURE);}
  if(pm.bet<=0){printf("ERROR: -b: Argument must be positive. Abort.\n");           exit(EXIT_FAILURE);}
  if(pm.eps< 0){printf("ERROR: -z: Argument must be in range [0,1]. Abort.\n");     exit(EXIT_FAILURE);}
  if(pm.eps> 1){printf("ERROR: -z: Argument must be in range [0,1]. Abort.\n");     exit(EXIT_FAILURE);}
  if(M<M0)     {printf("ERROR: -D: [Downsampling] M>M' is violated Abort.\n");      exit(EXIT_FAILURE);}
  if(N<N0)     {printf("ERROR: -D: [Downsampling] N>N' is violated Abort.\n");      exit(EXIT_FAILURE);}
  if(!strchr("exyn",pm.nrm)){printf("\n  ERROR: -u: Argument must be one of 'e', 'x', 'y' and 'n'. Abort.\n\n");exit(EXIT_FAILURE);}
}

void memsize(int *dsz, int *isz, pwsz sz, pwpm pm){
  int M=sz.M,N=sz.N,J=sz.J,K=sz.K,D=sz.D,Df=sz.Df,Dc=D+Df;
  int T=pm.opt&PW_OPT_LOCAL; int L=M+N,mtd=MAXTREEDEPTH;

  *isz =D;                *dsz =4*M+2*N+D*(5*M+N+14*D+3)+M*Df;    /* common            */
  *isz+=J?L:0;            *dsz+=J?J*(1+2*Dc+J):0;                 /* nystrom           */

  if(pm.opt&PW_OPT_NONRG) goto skip;
  *isz+=K?M:0;            *dsz+=K?K*(2*M+3*K+D+12):(3*M*M);       /* G: low/full rank  */
  skip:

  /* ssm */
  if(strlen(pm.fn[COV_LQ])) *dsz+=2*D*M*K;

  /* kdtree */
  if(!T) return;
  *isz+=(3*L+1)*2;                                                /* tree body (x2)    */
  *isz+=L*6;              *dsz+=2*L;                              /* work build        */
  *isz+=L*(2+mtd);                                                /* work eball/bbnext */

  /* normalized gauss prod */
  *dsz+=L*3*Dc;

}

void print_bbox(const double *X, int D, int N){
  int d,n; double max,min; char ch[3]={'x','y','z'};
  for(d=0;d<D;d++){
    max=X[d];for(n=1;n<N;n++) max=fmax(max,X[d+D*n]);
    min=X[d];for(n=1;n<N;n++) min=fmin(min,X[d+D*n]);
    fprintf(stderr,"%c=[%.2f,%.2f]%s",ch[d],min,max,d==D-1?"\n":", ");
  }
}

void print_norm(const double *X, const double *Y, int D, int N, int M, int sw, char type){
    int t=0; char name[4][64]={"for each","using X","using Y","skipped"};
    switch(type){
      case 'e': t=0; break;
      case 'x': t=1; break;
      case 'y': t=2; break;
      case 'n': t=3; break;
    }
    if(sw){
      fprintf(stderr,"  Normalization: [%s]\n",name[t]);
      fprintf(stderr,"    Bounding boxes that cover point sets:\n");
    }
    fprintf(stderr,"    %s:\n",sw?"Before":"After");
    fprintf(stderr,"      Target: "); print_bbox(X,D,N);
    fprintf(stderr,"      Source: "); print_bbox(Y,D,M);
    if(!sw) fprintf(stderr,"\n");
}

double tvcalc(const struct timeval *end, const struct timeval *beg){
  return (end->tv_sec-beg->tv_sec)+(end->tv_usec-beg->tv_usec)/1e6;
}

void fprint_comptime(FILE *fp, const struct timeval *tv, double *tt, int nx, int ny, int geok){
  if(fp==stderr) fprintf(fp,"  Computing Time:\n");
  #ifdef MINGW32
  if(geok)   fprintf(fp,"    FPSA algorithm:  %.3lf s\n", tvcalc(tv+2,tv+1));
  if(nx||ny) fprintf(fp,"    Downsampling:    %.3lf s\n", tvcalc(tv+3,tv+2));
  fprintf(fp,"    VB Optimization: %.3lf s\n",            tvcalc(tv+4,tv+3));
  if(ny)     fprintf(fp,"    Interpolation:   %.3lf s\n", tvcalc(tv+5,tv+4));
  #else
  if(geok)   fprintf(fp,"    FPSA algorithm:  %.3lf s (real) / %.3lf s (cpu)\n",tvcalc(tv+2,tv+1),(tt[2]-tt[1])/CLOCKS_PER_SEC);
  if(nx||ny) fprintf(fp,"    Downsampling:    %.3lf s (real) / %.3lf s (cpu)\n",tvcalc(tv+3,tv+2),(tt[3]-tt[2])/CLOCKS_PER_SEC);
  fprintf(fp,"    VB Optimization: %.3f s (real) / %.3lf s (cpu)\n",            tvcalc(tv+4,tv+3),(tt[4]-tt[3])/CLOCKS_PER_SEC);
  if(ny)     fprintf(fp,"    Interpolation:   %.3lf s (real) / %.3lf s (cpu)\n",tvcalc(tv+5,tv+4),(tt[5]-tt[4])/CLOCKS_PER_SEC);
  #endif
  fprintf(fp,"    File reading:    %.3lf s\n",tvcalc(tv+1,tv+0));
  fprintf(fp,"    File writing:    %.3lf s\n",tvcalc(tv+6,tv+5));
  if(fp==stderr) fprintf(fp,"\n");
}

void fprint_comptime2(FILE *fp, const struct timeval *tv, double *tt, int geok){
  fprintf(fp,"%lf\t%lf\n",tvcalc(tv+2,tv+1),(tt[2]-tt[1])/CLOCKS_PER_SEC);
  fprintf(fp,"%lf\t%lf\n",tvcalc(tv+3,tv+2),(tt[3]-tt[2])/CLOCKS_PER_SEC);
  fprintf(fp,"%lf\t%lf\n",tvcalc(tv+4,tv+3),(tt[4]-tt[3])/CLOCKS_PER_SEC);
  fprintf(fp,"%lf\t%lf\n",tvcalc(tv+5,tv+4),(tt[5]-tt[4])/CLOCKS_PER_SEC);
  fprintf(fp,"%lf\t%lf\n",tvcalc(tv+1,tv+0),tvcalc(tv+6,tv+5));
}

double* downsample_F (const double *F, int D, int L, const int *U, int num){
  int d,j; double *f=calloc(D*num,sizeof(double));
  for(j=0;j<num;j++)for(d=0;d<D;d++)f[d+D*j]=F[d+D*U[j]];
  return f;
}

double* downsample_LQ(const double *LQ, int M, int K, const int *U, int num, int D){
  int d,k,i; double *lq=NULL;
  if(!D)/*bcpd*/D=1;
  lq=calloc(K+D*num*K,sizeof(double));
  for(k=0;k<K;k++) lq[k]=LQ[k];
  for(k=0;k<K;k++)for(i=0;i<num;i++)for(d=0;d<D;d++) lq[d+D*i+D*num*k+K]=LQ[d+D*U[i]+D*M*k+K];
  return lq;
}

double *read_LQ(int *nr, int *nc, const char* file){
  int i,j,I,J; double *A,*B;
  A=read2dcm(&I,&J,file);
  B=calloc(I*J,sizeof(double));

  for(j=0;j<J;j++) B[j]=A[j];
  for(j=0;j<J;j++)for(i=0;i<I-1;i++) B[i+(I-1)*j+J]=A[j+J*i+J];

  free(A); *nr=I;*nc=J;
  return(B);
}

void print_pwpm_detailed(const pwpm* params) {
    if (params == NULL) {
        printf("错误：PWPM参数结构为NULL\n");
        return;
    }
    
    printf("\n==================== PWPM 详细参数说明 ====================\n");
    
    // 文件名信息
    printf("文件路径和名称:\n");
    char* fn_descriptions[8] = {
        "点集X文件", "点集Y文件", "法向量文件", "输出文件", 
        "权重文件", "参考文件", "模型文件", "日志文件"
    };
    
    for (int i = 0; i < 8; i++) {
        if (params->fn[i][0] != '\0') {
            printf("  %s: %s\n", fn_descriptions[i], params->fn[i]);
        }
    }
    
    // 标志和模式
    printf("\n基本设置和标志:\n");
    printf("  归一化模式 (nrm): %c", params->nrm);
    switch (params->nrm) {
        case 'n': printf(" (不归一化)\n"); break;
        case 'u': printf(" (单位归一化)\n"); break;
        case 's': printf(" (标准归一化)\n"); break;
        default: printf(" (未知模式)\n");
    }
    
    printf("  归一化标志 (nrf): %c", params->nrf);
    switch (params->nrf) {
        case '0': printf(" (禁用)\n"); break;
        case '1': printf(" (启用)\n"); break;
        default: printf(" (未知值)\n");
    }
    
    printf("  优化方法 (opt): %d", params->opt);
    switch (params->opt) {
        case 0: printf(" (非线性优化)\n"); break;
        case 1: printf(" (线性拉普拉斯优化)\n"); break;
        case 2: printf(" (混合优化)\n"); break;
        default: printf(" (未知优化方法)\n");
    }
    
    // 非线性优化参数
    printf("\n非线性优化参数 (BCPD/BCSPL相关):\n");
    printf("  最大迭代次数 (nlp): %d\n", params->nlp);
    printf("  控制点数量 (G): %d\n", params->G);
    printf("  噪声水平 (delta): %.6f\n", params->dlt);
    printf("  刚性项权重 (omega): %.6f\n", params->omg);
    printf("  正则化权重 (gamma): %.6f\n", params->gma);
    printf("  固定点权重 (fpv): %.6f\n", params->fpv);
    
    // 线性拉普拉斯参数
    printf("\n线性拉普拉斯优化参数:\n");
    printf("  最大迭代次数 (llp): %d\n", params->llp);
    printf("  拉普拉斯操作符尺寸 (J): %d\n", params->J);
    printf("  限制系数 (lim): %.6f\n", params->lim);
    printf("  拉格朗日乘子 (lambda): %.6f\n", params->lmd);
    printf("  平滑系数 (beta): %.6f\n", params->bet);
    
    // 随机搜索参数
    printf("\n随机搜索和初始化参数:\n");
    printf("  随机搜索次数 (rns): %d\n", params->rns);
    printf("  随机采样数 (K): %d\n", params->K);
    printf("  步长参数 (btn): %.6f\n", params->btn);
    printf("  缩放系数 (kappa): %.6f\n", params->kpa);
    printf("  收敛阈值 (cnv): %.6e\n", params->cnv);
    
    // 下采样参数
    printf("\n下采样参数:\n");
    printf("  点集X下采样数量 (dwn[0]): %d\n", params->dwn[0]);
    printf("  点集Y下采样数量 (dwn[1]): %d\n", params->dwn[1]);
    printf("  点集X下采样率 (dwr[0]): %.6f\n", params->dwr[0]);
    printf("  点集Y下采样率 (dwr[1]): %.6f\n", params->dwr[1]);
    
    // 近邻搜索参数
    printf("\n近邻搜索参数:\n");
    printf("  K近邻数量 (nnk): %d\n", params->nnk);
    printf("  近邻搜索半径 (nnr): %.6f\n", params->nnr);
    
    // 数值和收敛参数
    printf("\n数值计算和收敛参数:\n");
    printf("  τ参数 (tau): %.6f\n", params->tau);
    printf("  数值精度 (eps): %.6e\n", params->eps);
    
    printf("==================== PWPM 详细参数说明完成 ====================\n\n");
  }

// 为CFFI创建的wrapper函数

// Export wrapper for Python
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int bcpd_wrapper(
    const double *X_ptr, int N, int D,  // 目标点云
    const double *Y_ptr, int M,         // 源点云
    int random_seed,
    double omega, double lambda, double beta, double gamma, double kappa, // 模型参数
    int max_iter, int min_iter, double tolerance,  // 迭代参数
    double *y_out, double *v_out, double *s_out, double *R_out, double *t_out, // 输出变量
    double *sigma_out, int quiet_mode, // 其他选项参数
    int accel_flag, int nystrom_J, int nystrom_K, int kd_tree_flag, // 加速选项
    int gauss_kernel_type, double tau, int knn, double radius, // 核函数参数
    char normalization_type, // 归一化类型: 'e', 'x', 'y', 'n'
    int transformation_type, // 0: Tsrn, 1: Tan, 2: Ta, 3: Tsr, 4: Tr, 5: Tn
    double search_scale, double search_radius, double sigma_threshold,  // KD树参数
    int downsampling_flag, int dwn_X, int dwn_Y, double dwr_X, double dwr_Y, // 下采样参数
    int save_options // 添加的保存选项参数
) {
    int D_copy, M_copy, N_copy;
    int lp; // 迭代次数
    double s, r, Np; // 缩放系数、sigma、有效点数
    pwpm pm; 
    pwsz sz;
    
    double *x, *y, *a, *u, *v, *w, *R, *t, *sgm, *pf, *e;
    double *X_copy, *Y_copy, *wd, *LQ = NULL, *LQ0 = NULL;
    int *wi;
    
    double *x0 = NULL, *y0 = NULL, *v0 = NULL, *X0 = NULL, *Y0 = NULL;
    double *muX, *muY, sgmX, sgmY;
    double *fx = NULL, *fy = NULL, *fx0 = NULL, *fy0 = NULL, *mufx = NULL, *mufy = NULL;
    double sgmfx, sgmfy;
    
    int nx, ny, N0 = 0, M0 = 0;
    double rx, ry;
    int dsz, isz, ysz, xsz;
    int *Ux = NULL, *Uy = NULL;
    
    double tt[7]; 
    struct timeval tv[7];
    int sd = sizeof(double), si = sizeof(int);
    int geok = 0;
    
    // 初始化计时器
    gettimeofday(tv+0, NULL); 
    tt[0] = clock();
    
    // 初始化参数
    memset(&pm, 0, sizeof(pwpm));
    
    // 使用选项数组和pw_getopt_direct设置参数
    cmd_option* options;
    int num_options;
    
    // 构建选项数组，将所有参数转化为命令行选项形式
    build_bcpd_options(
        &options, &num_options,
        omega, lambda, beta, gamma, kappa,
        max_iter, min_iter, tolerance,
        nystrom_J, nystrom_K, kd_tree_flag, random_seed,
        1, quiet_mode, accel_flag,  // 默认使用历史模式
        normalization_type,
        gauss_kernel_type, tau, knn, radius,
        transformation_type,
        search_scale, search_radius, sigma_threshold,
        // 0x0240,  // 保存y和Y轨迹 (对应于 -s Y 选项)
        save_options,  // 使用传入的save_options
        dwn_X, dwn_Y, dwr_X, dwr_Y
    );
    // if(!(pm.opt&PW_OPT_QUIET)) printInfo(sz,pm);
    // 使用直接选项设置参数
    pw_getopt_direct(&pm, options, num_options);
    // printf("pm->btn: %f\n", pm.btn);
    // 释放选项数组
    // free_options(options, num_options);
   
    // 设置输出前缀为空，因为我们不需要文件输出
    strcpy(pm.fn[OUTPUT], "");
    
    // 设置大小
    sz.N = N;
    sz.M = M;
    sz.D = D;
    sz.Df = 0; // 无函数值
    sz.J = pm.J;
    sz.K = pm.K;

    // /* read covariance (case: ssm) */
    // ssm=strlen(pm.fn[COV_LQ]);
    // if(ssm){
    //     LQ=read_LQ(&nr,&nc,pm.fn[COV_LQ]); pm.K=K=nc;
    //     if(nr!=1+D*M){printf("ERROR: The size of LQ is incosistent with that of X and Y.\n"); exit(EXIT_FAILURE);}
    // }

    /* init: random number */
    init_genrand64(pm.rns?pm.rns:clock());

    // 拷贝输入点云数据
    N_copy = N;
    M_copy = M;
    D_copy = D;
    
    // // 创建数组副本以防止修改原始数据
    X_copy = (double*)malloc(D * N * sizeof(double));
    Y_copy = (double*)malloc(D * M * sizeof(double));
    
    if (!X_copy || !Y_copy) {
        fprintf(stderr, "内存分配失败\n");
        if (X_copy) free(X_copy);
        if (Y_copy) free(Y_copy);
        return -4;
    }
    
    // CFFI传入的数组是行主序，需要转换为列主序
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X_copy[d + D * n] = X_ptr[n * D + d];
        }
    }
    
    for (int m = 0; m < M; m++) {
        for (int d = 0; d < D; d++) {
            Y_copy[d + D * m] = Y_ptr[m * D + d];
        }
    }

    // X_copy = X_ptr; // 不需要复制，直接使用原始指针
    // Y_copy = Y_ptr; // 不需要复制，直接使用原始指针
    
    gettimeofday(tv+1, NULL); 
    tt[1] = clock();

    // /* print: paramters */
    // if(!(pm.opt&PW_OPT_QUIET)) printInfo(sz,pm);
    
    // 检查参数
    check_prms(pm, sz);

    /* print: paramters */
    if(!(pm.opt&PW_OPT_QUIET)) printInfo(sz,pm);
    
    // 余下部分代码保持不变...
    // 归一化
    muX = calloc(D, sd); 
    muY = calloc(D, sd);
    
    if (!(pm.opt & PW_OPT_QUIET) && (D == 2 || D == 3)) 
        print_norm(X_copy, Y_copy, D, N, M, 1, pm.nrm);
    
    normalizer(muX, &sgmX, muY, &sgmY, X_copy, Y_copy, N, M, D, pm.nrm);
    normalize(X_copy, muX, sgmX, N, D);
    normalize(Y_copy, muY, sgmY, M, D);
    
    if (!(pm.opt & PW_OPT_QUIET) && (D == 2 || D == 3)) 
        print_norm(X_copy, Y_copy, D, N, M, 0, pm.nrm);
    
    // 测地线核函数计算
    gettimeofday(tv+2, NULL); 
    tt[2] = clock();
    
    if (!(pm.opt & PW_OPT_NONRG)) {
        geok = (pm.nnk || strlen(pm.fn[FACE_Y])) && pm.tau > 1e-5;
        if (geok && !(pm.opt & PW_OPT_QUIET)) 
            fprintf(stderr, "  Executing the FPSA algorithm ... ");
        
        if (geok) { 
            sgraph* sg;
            int K;
            if (pm.nnk) 
                sg = sgraph_from_points(Y_copy, D, M, pm.nnk, pm.nnr);
            else       
                sg = sgraph_from_mesh(Y_copy, D, M, pm.fn[FACE_Y]);
            
            LQ = geokdecomp(&K, Y_copy, D, M, (const int**)sg->E, (const double**)sg->W, pm.K, pm.bet, pm.tau, pm.eps);
            sz.K = pm.K = K; // 更新K
            sgraph_free(sg);
            
            if (geok && !(pm.opt & PW_OPT_QUIET)) 
                fprintf(stderr, "done. (K->%d)\n\n", K);
        }
    }
    
    // 下采样
    gettimeofday(tv+3, NULL); 
    tt[3] = clock();
    
    nx = pm.dwn[TARGET]; 
    rx = pm.dwr[TARGET];
    ny = pm.dwn[SOURCE]; 
    ry = pm.dwr[SOURCE];
    
    if ((nx || ny) && !(pm.opt & PW_OPT_QUIET)) 
        fprintf(stderr, "  Downsampling ...");
    
    if (nx) {
        X0 = X_copy; 
        N0 = N; 
        N = sz.N = nx;
        X_copy = calloc(D * N, sd);
        Ux = calloc(rx == 0 ? N0 : N, si);
        downsample(X_copy, Ux, N, X0, D, N0, rx);
    }
    
    if (ny) {
        Y0 = Y_copy; 
        M0 = M; 
        M = sz.M = ny;
        Y_copy = calloc(D * M, sd);
        Uy = calloc(ry == 0 ? M0 : M, si);
        downsample(Y_copy, Uy, M, Y0, D, M0, ry);
        
        if (LQ) {
            LQ0 = LQ;
            LQ = downsample_LQ(LQ0, M0, pm.K, Uy, M, 0);
        }
    }
    
    if ((nx || ny) && !(pm.opt & PW_OPT_QUIET)) 
        fprintf(stderr, " done. \n\n");
    
    gettimeofday(tv+4, NULL); 
    tt[4] = clock();
    
    // 分配内存
    memsize(&dsz, &isz, sz, pm);
    ysz = D * M; 
    ysz+=D*M*((pm.opt&PW_OPT_PATHY)?pm.nlp:0);
    xsz = D * M; 
    xsz+=D*M*((pm.opt&PW_OPT_PATHX)?pm.nlp:0);
    
    wd = calloc(dsz, sd); 
    x = calloc(xsz, sd); 
    a = calloc(M, sd); 
    u = calloc(D * M, sd); 
    R = calloc(D * D, sd); 
    sgm = calloc(M, sd);
    wi = calloc(isz, si); 
    y = calloc(ysz, sd); 
    w = calloc(M, sd); 
    v = calloc(D * M, sd); 
    t = calloc(D, sd); 
    pf = calloc(3 * pm.nlp, sd);
    e = fx ? calloc(sz.Df, sd) : NULL;

    // print_pwpm_detailed(&pm);
    
    // 主要计算
    lp = bcpd(x, y, u, v, w, a, sgm, &s, R, t, &r, e, &Np, pf, wd, wi, X_copy, Y_copy, fx, fy, LQ, sz, pm);

    /* save correspondence */
    if((pm.opt&PW_OPT_SAVEP)|(pm.opt&PW_OPT_SAVEC)|(pm.opt&PW_OPT_SAVEE))
    if(!(nx||ny)) save_corresp(pm.fn[OUTPUT],X_copy,y,a,sgm,s,r,sz,pm);
    /* save trajectory */
    char *ytraj=".optpath.bin",*xtraj=".optpathX.bin";
    if(pm.opt&PW_OPT_PATHX) save_optpath(xtraj,x+D*M,X_copy,sz,pm,lp);
    if(pm.opt&PW_OPT_PATHY) save_optpath(ytraj,y+D*M,X_copy,sz,pm,lp);
    
    // printf("start 插值\n");
    // 插值
    gettimeofday(tv+5, NULL); 
    tt[5] = clock();
    
    if (ny) {
        if (!(pm.opt & PW_OPT_QUIET)) 
            fprintf(stderr, "%s  Interpolating ... ", (pm.opt & PW_OPT_HISTO) ? "\n" : "");
        
        y0 = calloc(D * M0, sd); 
        v0 = calloc(D * M0, sd); 
        x0 = (pm.opt & PW_OPT_SAVEX) ? calloc(D * M0, sd) : NULL;
        
        if (LQ0) 
            interpolate2(y0, v0, Y0, M0, x, Y_copy, w, &s, R, t, &r, LQ0, Uy, sz, pm);
        else     
            interpolate1(y0, v0, Y0, M0, x, Y_copy, w, &s, R, t, &r, sz, pm);
        
        if (x0)  
            interpolate_x(x0, y0, X0, D, M0, N0, r, pm);
        
        if (!(pm.opt & PW_OPT_QUIET)) 
            fprintf(stderr, "done. \n\n");
        
        // 交换
        SWAP(M, M0); 
        
        if (X0) { free(X_copy); X_copy = X0; X0 = NULL; }
        if (y0) { free(y); y = y0; y0 = NULL; }
        if (v0) { free(v); v = v0; v0 = NULL; }
        
        SWAP(N, N0); 
        
        if (Y0) { free(Y_copy); Y_copy = Y0; Y0 = NULL; }
        if (x0) { free(x); x = x0; x0 = NULL; }
    }
    
    gettimeofday(tv+6, NULL); 
    tt[6] = clock();
    
    // printf("start 去归一化\n");
    // 去归一化 (y,v,s,R,t)
    denormalize(y, muX, sgmX, M, D);
    
    {
        int d, i, m;
        double val;
        
        s = (sgmX / sgmY) * s;
        for (m = 0; m < M; m++) 
            for (d = 0; d < D; d++) 
                v[d + D * m] *= sgmY;
        
        for (d = 0; d < D; d++) 
            t[d] = sgmX * t[d] + muX[d];
        
        for (d = 0; d < D; d++) {
            val = 0;
            for (i = 0; i < D; i++) {
                val += R[d + D * i] * muY[i];
            } 
            t[d] -= s * val;
        }
        
        if (pm.opt & PW_OPT_AFFIN) 
            for (d = 0; d < D; d++)
                for (i = 0; i < D; i++) 
                    R[d + D * i] *= s;
    }
    
    // printf("start 将结果复制到输出参数\n");
    // 将结果复制到输出参数
    if (y_out != NULL) {
        for (int m = 0; m < M; m++) {
            for (int d = 0; d < D; d++) {
                y_out[m * D + d] = y[d + D * m]; // 列主序转为行主序
            }
        }
    }
    
    if (v_out != NULL) {
        for (int m = 0; m < M; m++) {
            for (int d = 0; d < D; d++) {
                v_out[m * D + d] = v[d + D * m]; // 列主序转为行主序
            }
        }
    }
    
    if (s_out != NULL) {
        *s_out = s;
    }
    
    if (R_out != NULL) {
        for (int d1 = 0; d1 < D; d1++) {
            for (int d2 = 0; d2 < D; d2++) {
                R_out[d1 * D + d2] = R[d2 + D * d1]; // 列主序转为行主序
            }
        }
    }
    
    if (t_out != NULL) {
        for (int d = 0; d < D; d++) {
            t_out[d] = t[d];
        }
    }
    
    if (sigma_out != NULL) {
        *sigma_out = r;
    }
    // printf("start 释放内存\n");
    // 释放内存
    if (wd) free(wd);
    if (wi) free(wi);
    if (x) free(x);
    if (y) free(y);
    if (a) free(a);
    if (u) free(u);
    if (v) free(v);
    if (w) free(w);
    if (R) free(R);
    if (t) free(t);
    if (sgm) free(sgm);
    if (pf) free(pf);
    if (muX) free(muX);
    if (muY) free(muY);
    
    if (X_copy) free(X_copy);
    if (Y_copy) free(Y_copy);
    if (X0) free(X0);
    if (Y0) free(Y0);
    if (LQ) free(LQ);
    if (LQ0) free(LQ0);
    if (e) free(e);
    if (Ux) free(Ux);
    if (Uy) free(Uy);
    
    // 返回迭代次数
    // printf("返回迭代次数\n");
    return lp;
}

int main(int argc, char** argv) {
    // 从命令行参数解析：
    // -x X -y Y -w 0.0 -b 2.0 -l 20.0 -g 10 -J 300 -K 70 -p -c 1e-6 -n 500 -h -r 1 -s Y
    
    // 设置参数
    const char* target_file = "X";  // -x X
    const char* source_file = "Y";  // -y Y
    double omega = 0.0;        // -w 0.0
    double beta = 2.0;         // -b 2.0
    double lambda = 20.0;      // -l 20.0
    double gamma = 10.0;       // -g 10
    int nystrom_J = 300;       // -J 300
    int nystrom_K = 70;        // -K 70
    int kd_tree = 1;           // -p
    double tolerance = 1e-6;   // -c 1e-6
    int max_iter = 500;        // -n 500
    int history_mode = 1;      // -h
    int random_seed = 1;       // -r 1
    int save_trajectory = 1;   // -s Y

    // 其他必需参数（未在命令行中指定，使用默认值）
    double kappa = 0;      // 默认值（无穷大）
    int min_iter = 30;         // 默认值
    int quiet_mode = 0;        // 默认不安静模式
    int accel_flag = 0;        // 默认不用加速（因为已经明确指定了J和K）
    int gauss_kernel_type = 0; // 默认高斯核
    double tau = 0.0;          // 默认值
    int knn = 0;               // 默认值
    double radius = 0.0;       // 默认值
    char normalization_type = 'e'; // 默认分别归一化
    int transformation_type = 0;   // 默认sR(y+v)+t
    double search_scale = 7.0;     // 默认值
    double search_radius = 0.15;   // 默认值
    double sigma_threshold = 0.2;  // 默认值
    int downsampling_flag = 0;     // 默认不进行下采样
    int dwn_X = 0, dwn_Y = 0;      // 默认值
    double dwr_X = 0.0, dwr_Y = 0.0; // 默认值

    // 读取点云数据
    int N, M, D;
    double *X_col, *Y_col; // 列主序数据
    double *X_row, *Y_row; // 行主序数据
    
    printf("读取目标点云: %s\n", target_file);
    X_col = read2dcm(&N,&D,"/home/han/SSF/pybcpd/src/pybcpd/register/tgt.txt"); 
    if (!X_col) {
        fprintf(stderr, "无法读取目标点云文件: %s\n", target_file);
        return -1;
    }
    printf("读取源点云: %s\n", source_file);
    Y_col = read2dcm(&M,&D,"/home/han/SSF/pybcpd/src/pybcpd/register/src.txt");
    if (!Y_col) {
        fprintf(stderr, "无法读取源点云文件: %s\n", source_file);
        free(X_col);
        return -1;
    }
    
    // // 将数据从列主序转换为行主序
    X_row = (double*)malloc(N * D * sizeof(double));
    Y_row = (double*)malloc(M * D * sizeof(double));
    
    for (int n = 0; n < N; n++) {
        for (int d = 0; d < D; d++) {
            X_row[n * D + d] = X_col[d * N + n];
        }
    }
    
    for (int m = 0; m < M; m++) {
        for (int d = 0; d < D; d++) {
            Y_row[m * D + d] = Y_col[d * M + m];
        }
    }
    
    printf("目标点云: %d 点, 源点云: %d 点, 维度: %d\n", N, M, D);
    
    // 输出参数
    double* y_out = (double*)malloc(M * D * sizeof(double)); // 变换后的点云
    double* v_out = (double*)malloc(M * D * sizeof(double)); // 位移向量
    double s_out = 0.0;                                      // 缩放因子
    double* R_out = (double*)malloc(D * D * sizeof(double)); // 旋转矩阵
    double* t_out = (double*)malloc(D * sizeof(double));     // 平移向量
    double sigma_out = 0.0;                                  // 最终的sigma值
    
    // 显示参数
    printf("\nBCPD/DET/DLD version 0.95.0 测试\n");
    printf("\n参数设置:\n");
    printf("  target file = %s\n", target_file);
    printf("  source file = %s\n", source_file);
    printf("  omega  = %.2f\n", omega);
    printf("  lambda = %.2f\n", lambda);
    printf("  beta   = %.2f\n", beta);
    printf("  gamma  = %.2f\n", gamma);
    printf("  tolerance = %.1e\n", tolerance);
    printf("  max_iter = %d\n", max_iter);
    printf("  Nystrom J = %d, K = %d\n", nystrom_J, nystrom_K);
    printf("  KD-Tree: %s\n", kd_tree ? "开启" : "关闭");
    printf("  历史模式: %s\n", history_mode ? "开启" : "关闭");
    printf("  随机种子: %d\n", random_seed);
    printf("  轨迹保存: %s\n", save_trajectory ? "是" : "否");
    
    // 计时
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);
    
    // 调用BCPD包装函数
    printf("\n执行BCPD算法...\n");
    int iterations = bcpd_wrapper(
        X_col, N, D,
        Y_col, M,
        random_seed,
        omega, lambda, beta, gamma, kappa,
        max_iter, min_iter, tolerance,
        y_out, v_out, &s_out, R_out, t_out,
        &sigma_out, quiet_mode,
        accel_flag, nystrom_J, nystrom_K, kd_tree,
        gauss_kernel_type, tau, knn, radius,
        normalization_type,
        transformation_type,
        search_scale, search_radius, sigma_threshold,
        downsampling_flag, dwn_X, dwn_Y, dwr_X, dwr_Y,
        0x0240
    );
    
    gettimeofday(&end_time, NULL);
    double run_time = (end_time.tv_sec - start_time.tv_sec) + 
                     (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
    
    // 显示结果
    printf("\nBCPD完成，用时: %.3f 秒\n", run_time);
    printf("迭代次数: %d\n", iterations);
    printf("最终sigma: %f\n", sigma_out);
    printf("缩放因子: %f\n", s_out);
    
    // 显示旋转矩阵
    printf("\n旋转矩阵 (%d x %d):\n", D, D);
    for (int i = 0; i < D; i++) {
        printf("  ");
        for (int j = 0; j < D; j++) {
            printf("%8.4f ", R_out[i * D + j]);
        }
        printf("\n");
    }
    
    // 显示平移向量
    printf("\n平移向量 (1 x %d):\n  ", D);
    for (int d = 0; d < D; d++) {
        printf("%8.4f ", t_out[d]);
    }
    printf("\n");
    
    // 显示前几个点的变换结果
    int show_points = 5 < M ? 5 : M;
    printf("\n前%d个变换后的点:\n", show_points);
    for (int i = 0; i < show_points; i++) {
        printf("  y[%d] = (", i);
        for (int d = 0; d < D; d++) {
            printf("%f%s", y_out[i * D + d], d == D - 1 ? ")\n" : ", ");
        }
    }
    
    // 保存结果到文件
    printf("\n保存结果到文件...\n");
    
    FILE* fp;
    
    // 保存变换后的点云
    fp = fopen("output_y.txt", "w");
    if (fp) {
        for (int m = 0; m < M; m++) {
            for (int d = 0; d < D; d++) {
                fprintf(fp, "%lf%c", y_out[m * D + d], d == D - 1 ? '\n' : '\t');
            }
        }
        fclose(fp);
        printf("  变换后点云: output_y.txt\n");
    }
    
    // 保存位移向量
    fp = fopen("output_v.txt", "w");
    if (fp) {
        for (int m = 0; m < M; m++) {
            for (int d = 0; d < D; d++) {
                fprintf(fp, "%lf%c", v_out[m * D + d], d == D - 1 ? '\n' : '\t');
            }
        }
        fclose(fp);
        printf("  位移向量: output_v.txt\n");
    }
    
    // 保存旋转矩阵
    fp = fopen("output_R.txt", "w");
    if (fp) {
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) {
                fprintf(fp, "%lf%c", R_out[i * D + j], j == D - 1 ? '\n' : '\t');
            }
        }
        fclose(fp);
        printf("  旋转矩阵: output_R.txt\n");
    }
    
    // 保存平移向量
    fp = fopen("output_t.txt", "w");
    if (fp) {
        for (int d = 0; d < D; d++) {
            fprintf(fp, "%lf%c", t_out[d], d == D - 1 ? '\n' : '\t');
        }
        fclose(fp);
        printf("  平移向量: output_t.txt\n");
    }
    
    // 清理内存
    if (X_col) free(X_col);
    if (Y_col) free(Y_col);
    if (X_row) free(X_row);
    if (Y_row) free(Y_row);
    if (y_out) free(y_out);
    if (v_out) free(v_out);
    if (R_out) free(R_out);
    if (t_out) free(t_out);
    
    printf("\n测试完成\n");
    return 0;
}