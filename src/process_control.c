#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

int main() {
    printf("=== Process Lifecycle Demo ===\n");
    printf("[PARENT] Running with PID: %d\n\n", getpid());
    printf("[PARENT] Spawning child and waiting...\n\n");

    pid_t pid = fork();

    if (pid < 0) {
        perror("Fork failed");
        exit(1);
    } else if (pid == 0) {
        printf("[CHILD] Created successfully.\n");
        printf("[CHILD] Child PID: %d\n", getpid());
        printf("[CHILD] Parent PPID: %d\n", getppid());
        printf("[CHILD] Executing 'ls -la' via execlp()...\n");
        execlp("ls", "ls", "-la", NULL);
        perror("Exec failed");
        exit(1);
    } else {
        int status;
        wait(&status);
        printf("\n[CHILD] Execution finished.\n");
        printf("[PARENT] Child exited, terminating parent process.\n");
    }
    return 0;
}
