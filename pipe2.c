#include<stdio.h>
#include<unistd.h>
int main(){
int fd[2];
int pid = fork();
pipe(fd);
char buf[10]="hello";
if(pid>0){
close(fd[0]);
write(fd[1],buf,sizeof(buf));
}
if(pid==0){
close(fd[0]);
read(fd[0],buf,sizeof(buf));
}}
