#include <stdio.h>

int func(int i) {
    if (i % 2)  // This checks if 'i' is odd
        return 0;
    else
        return i;
}

int main() {
    int i = 3;
    i = func(i);
    i = func(i);
    printf("%d mystery", i);  // This will print the result after calling func twice
    return 0;
}
