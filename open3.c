#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>
int main(){
int fd1,fd2;
size_t n;
char buf[100];
fd1=open("f101",O_WRONLY);
fd2=open("f102",O_WRONLY);
printf("opened\n");
printf("%d\n",fd1);
printf("%d\n",fd2);
n=read(fd1,buf,30);
write(fd2,buf,n);
}
