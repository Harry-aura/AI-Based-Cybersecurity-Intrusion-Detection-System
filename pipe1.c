#include <unistd.h>
#include <stdio.h>
int main(){
int fd[2];
if(pipe(fd)==-1)
printf("pipe is not successfull\n");
else {
printf("read end of the pipe is %d\n",fd[0]);
printf("write end of the pipe is %d\n",fd[1]);

}}
