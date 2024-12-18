import graphviz

dot = graphviz.Digraph(comment='CNN Architecture Diagram')

# Input layer
dot.node('input', 'Input (32x32 grayscale image)')

# Convolutional layer 1
dot.node('conv1', 'Conv2D (32 filters, 3x3 kernel, ReLU activation)')
dot.edge('input', 'conv1')

# Max pooling layer 1
dot.node('pool1', 'MaxPooling2D (2x2 pool size)')
dot.edge('conv1', 'pool1')

# Convolutional layer 2
dot.node('conv2', 'Conv2D (64 filters, 3x3 kernel, ReLU activation)')
dot.edge('pool1', 'conv2')

# Max pooling layer 2
dot.node('pool2', 'MaxPooling2D (2x2 pool size)')
dot.edge('conv2', 'pool2')

# Flatten layer
dot.node('flatten', 'Flatten')
dot.edge('pool2', 'flatten')

# Dense layer 1
dot.node('dense1', 'Dense (128 neurons, ReLU activation)')
dot.edge('flatten', 'dense1')

# Dropout layer
dot.node('dropout', 'Dropout (0.5 rate)')
dot.edge('dense1', 'dropout')

# Dense layer 2
dot.node('dense2', 'Dense (7 neurons, softmax activation)')
dot.edge('dropout', 'dense2')

# Output layer
dot.node('output', 'Output (7-class classification)')
dot.edge('dense2', 'output')

dot.render('cnn_architecture_diagram', view=True)
