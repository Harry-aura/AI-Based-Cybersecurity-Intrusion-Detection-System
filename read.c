#include <fcntl.h>
#include <unistd.h>
int main(){
char buf[100]; 
int fd1=open("f1",O_RDONLY);
int fd2=open("f2",O_WRONLY | O_CREAT,0644);
int Read=read(fd1,buf,20);
write(fd2,buf,Read);
}
