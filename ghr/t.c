#include <stdio.h>

int main() {
    int n = 2;
    int sum = 5;

    switch (n) {
        case 2:
            sum -= 2;  // Fall through intentionally
        case 3:
            sum *= 5;
            break;
        default:
            sum = 0;
    }

    printf("The value of sum is: %d\n", sum);
    return 0;
}
