#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
int main(){
int fd1,fd2;

char buf[100];
size_t n;
fd1=open("f1",O_RDONLY);
fd2=open("b2",O_WRONLY,0655);
while((n=read(fd1,buf,sizeof(buf)))>0);
write(fd2,buf,n);
printf("%s",buf);


}
