#include <stdio.h>
#include <unistd.h>
int main() {
int fd[2];
int n=10;
pipe(fd);
int pid=fork();
if(pid>0) {
close(fd[0]);
write(fd[1],&n,sizeof(n));
}
if(pid==0){
close(fd[1]);
read(fd[0],&n,sizeof(n));
printf("square%d\n",n*n);
}


}
