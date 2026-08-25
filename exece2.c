#include<stdio.h>
#include<unistd.h>
int main(){
execlp("ls","ls","-F",NULL);
}
