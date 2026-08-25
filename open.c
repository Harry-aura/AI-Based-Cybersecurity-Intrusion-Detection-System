#include<stdio.h>
#include<fcntl.h>
#include<unistd.h>
int main(){
int fd;
char buf[100];
size_t n;
fd=open("f4",O_APPEND);
printf("%d\n",fd);
if(fd==-1)printf("error cannot open the file\n");
else printf("success\n");

n=write(fd,buf,100);
printf("%ld\n",n);
printf("%s\n",buf);
}






