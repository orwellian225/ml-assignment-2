#include <iostream>

#include <fmt/core.h>

#include "neuralnet.h"

/* NOTE TO SELF ON CMAKE:
 *  Add new .cpp files to the cmake file and then rebuild the make file
*/

int main() {

    nnlayer_t test_layer = init_layer(5, 2);
    test_layer.print();

    return 0;
}