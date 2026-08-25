#include<unistd.h>
#include<stdio.h>
int main(){
int fd[2];
pipe(fd);
int pid=fork();
if(pid>0){
close(fd[0]);
dup2(fd[1],1);
execlp("cat","cat","class.c",NULL);}
if(pid==0){
close(fd[1]);
dup2(fd[0],0);
execlp("wc","wc","-l","-c",NULL);}


}
