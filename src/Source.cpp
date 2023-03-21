#include "MovingWindow.h"

int main()
{
    MovingWindow window(1920, 1080, "Random Agent Simulation", 10000);
    window.run();
    return 0;
}