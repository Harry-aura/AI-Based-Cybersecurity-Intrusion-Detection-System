#include<stdio.h>
#include<unistd.h>
int main(){
char *args[]={"pwd",NULL};
execvp("pwd",args);
perror("not successfull");

}
