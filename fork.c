#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
int main(){
int pid;
pid=fork();
if(pid==0){
printf("child process\n");
printf("child process id %d\n",getpid());
printf("parent process id %d\n",getppid());

}
else if(pid>0){
printf("parent process\n");
printf("parent process id %d\n",getpid());
printf("parent process id %d\n",getppid());}
else {
printf("failure\n");}

}
