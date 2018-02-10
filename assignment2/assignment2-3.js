
// This is the script for 6.s198 Assignments 2 and 3.

console.log('Starting script');
console.log('Importing and initializing the dataset');

const xhrDatasetConfigs = {
  MNIST: {
    data: [{
      name: 'images',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png',
      dataType: 'png',
      shape: [28, 28, 1],
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8',
      dataType: 'uint8',
      shape: [10],
    }],
  },

  Fashion_MNIST: {
    data: [{
      name: 'images',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_images.png',
      dataType: 'png',
      shape: [28, 28, 1],
    }, {
      name: 'labels',
      path: 'https://storage.googleapis.com/learnjs-data/model-builder/fashion_mnist_labels_uint8',
      dataType: 'uint8',
      shape: [10],
    }],
  },

  CIFAR_10: {
    "data": [{
      "name": "images",
      "path": "https://storage.googleapis.com/learnjs-data/model-builder/cifar10_images.png",
      "dataType": "png",
      "shape": [32, 32, 3]
    }, {
      "name": "labels",
      "path": "https://storage.googleapis.com/learnjs-data/model-builder/cifar10_labels_uint8",
      "dataType": "uint8",
      "shape": [10]
    }],
    },
}

const dataSets = {};

function populateDatasets() {
  for(const datasetName in xhrDatasetConfigs) {
    if (xhrDatasetConfigs.hasOwnProperty(datasetName)) {
      dataSets[datasetName] =
        new dl.XhrDataset(xhrDatasetConfigs[datasetName]);
    }
  }
}

populateDatasets();

// To change between MNIST and CFAR, change the definitions
// of dataSet and showColor by commenting and uncommenting the 
// lines below, and reload the page

// use these two lines for MNIST
const dataSet = dataSets.MNIST;
const showColor = false;

// use these two lines for Fashion MNIST
//const dataSet = dataSets.Fashion_MNIST;
//const showColor = false;

// use these two lines for CIFAR
//const dataSet = dataSets.CIFAR_10;
//const showColor = true;

// functions for setting up the training data and the test data

const TRAIN_TEST_RATIO = 5 / 6;

function getTrainingData() {
  const [images, labels] = dataSet.getData();
  const end = Math.floor(TRAIN_TEST_RATIO * images.length);
  return [images.slice(0, end), labels.slice(0, end)];
}

function getTestData() {
  const data = dataSet.getData();
  if (data == null) { return null; }
  const [images, labels] = dataSet.getData();
  const start = Math.floor(TRAIN_TEST_RATIO * images.length);
  return [images.slice(start), labels.slice(start)];
}

console.log('DataSet initialized.  Please wait ...')


// Procedures to construct networks by adding layers to existing networks

// Add a flatten layer
function addFlattenLayer(graph, previousLayer) {
  return graph.reshape(previousLayer, [dl.util.sizeFromShape(previousLayer.shape)]);
}

// Add a convolutional layer with specified field size, stride, zero padding, and output depth
// Index is used for naming the layer

function addConvLayer(graph, previousLayer, index, fieldSize, stride, zeroPad, outputDepth) {
  inputShape = previousLayer.shape;

  console.log("adding convLayer");
  console.log('inputShape:', inputShape.toString());
  const wShape = [fieldSize, fieldSize, inputShape[2], outputDepth];
  console.log('layerspec:', wShape.toString());

  w = dl.Array4D.randTruncatedNormal(wShape, 0, 0.1);
  b = dl.Array1D.zeros([outputDepth]);

  const wTensor = graph.variable(`conv2d-${index}-w`, w);
  const bTensor = graph.variable(`conv2d-${index}-b`, b);

  return graph.conv2d(previousLayer, wTensor, bTensor, fieldSize, outputDepth, stride, zeroPad);
}

// add a Rectified Linear Unit layer

function addReluLayer(graph, previousLayer){
  return graph.relu(previousLayer);
}

// add a max pool layer with specified field size, stide, and zero padding

function addMaxPoolLayer(graph, previousLayer, fieldSize, stride, zeroPad){
  return graph.maxPool(previousLayer, fieldSize, stride, zeroPad);
}

// Add a fully connected layer with a specified number of hidden units
// The index here is used for naming the layer

function addFcLayer(graph, previousLayer, index, hiddenUnits) {
  const weightsInitializer = new dl.VarianceScalingInitializer();
  const biasInitializer = new dl.ZerosInitializer();
  const useBias = true;
  return graph.layers.dense(
    `fc-${index}`, previousLayer, hiddenUnits, null, useBias, weightsInitializer, biasInitializer);
}


// Hyperparameters to experiment with
const batchSize = 20;
const NUM_BATCHES = 50;

// print training results every batchInterval number of batches
const batchInterval = 20;

// flag to control whether or not to do testing
const DO_TESTING = false;

const NUM_IMAGES_TO_TEST = 10;

const CIFAR_10_LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

// Glabal variable that can be examined using the Javascript console
var LABELS_TO_EXAMINE = [];
var IMAGES_TO_EXAMINE = [];

// Helper procedure for viewing the images
// These will appear on the HTML page

// The images have very low resolution, so this does not produce good views
function showimage(num) {
  renderToCanvas(IMAGES_TO_EXAMINE[num], document.getElementById("imageHolder"));
}


//function renderToCanvas(a: Array3D, canvas: HTMLCanvasElement) {
function renderToCanvas(a, canvas) {
  const [height, width, depth] = a.shape;
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  const imageData = new ImageData(width, height);
  const data = a.dataSync();
  for (let i = 0; i < height * width; ++i) {
    const j = i * 4;
    const k = i * depth;
    if (showColor) {
      imageData.data[j + 0] = Math.round(255 * data[k + 0]);
      imageData.data[j + 1] = Math.round(255 * data[k + 1]);
      imageData.data[j + 2] = Math.round(255 * data[k + 2]);
      imageData.data[j + 3] = 255;
      } else {
	const pixel = Math.round(255 * data[k]);
	imageData.data[j+0] = pixel;
	imageData.data[j+1] = pixel;
	imageData.data[j+2] = pixel;
	imageData.data[j+3] = 255;
      }
      ctx.putImageData(imageData, 0, 0);
  }
}

// Helper procedures
function indexOfMax(a) {
  return a.indexOf(a.reduce((max, value) => {return Math.max(max, value)}));
  }
			    
function labelTag(n) {
  if (showColor) {
    return CIFAR_10_LABELS[n];
    } else {
      return n;
    }
}

function decimalToPercent(num) {
  return Math.floor(num * 100).toString() + '%';
  }

//  Here is the function that builds the network and does the training and testing.
// Note that the variables defined in the body of runModel
// local to the function activation, so you cannot directly examine them in the console.

function runModel() {

  console.log('Building the model');

  // build the model
  const graph = new dl.Graph();

  const inputShape = dataSet.getDataShape(0);
  const inputLayer = graph.placeholder('input', inputShape);
  const labelShape = dataSet.getDataShape(1);
  const labelTensor = graph.placeholder('label', labelShape);

  const flattenLayer = addFlattenLayer(graph, inputLayer);
  const fcLayer1 = addFcLayer(graph, flattenLayer, 1, 10);

  // This is a very simple network.  Here's where you should
  // include code to create new layers.
  // Don't forget to change outputLayer and costTensor to account for the 
  // new layers

  const outputLayer = graph.softmax(fcLayer1);

  // use cross entropy to compute the overall error ("cost") to be minimized
  const costTensor = graph.softmaxCrossEntropyCost(fcLayer1, labelTensor);

  const math = dl.ENV.math;
  const session = new dl.Session(graph, math);
  const learningRate = 0.1;

  // The optimizer here is stochastic gradient descent (SGD)
  const optimizer = new dl.SGDOptimizer(learningRate);

  const trainingData = getTrainingData();

  const trainingShuffledInputProviderGenerator =
    new dl.InCPUMemoryShuffledInputProviderBuilder(trainingData);
  const [trainInputProvider, trainLabelProvider] =
    trainingShuffledInputProviderGenerator.getInputProviders();

  const trainFeeds = [
    { tensor: inputLayer, data: trainInputProvider },
    { tensor: labelTensor, data: trainLabelProvider },
  ];

  console.log('Model built');

  math.scope(() => {
  console.log('begin training ', NUM_BATCHES, ' batches, will print progress every ', batchInterval, ' batches');
    trainStart = performance.now();
    for (let i = 0; i < NUM_BATCHES; i += 1) {
      const cost = session.train(
        costTensor, trainFeeds, batchSize, optimizer, dl.CostReduction.MEAN);

      // Compute the cost (by calling get), which requires transferring data
      // from the GPU.
      // save the cost for later examination
      // print a message every 50 batches
      if (!(i%batchInterval)){
        console.log('batch', i, '---', 'Cost:', cost.get());
      }
      lastAverageCost = cost.get();
 }
    trainEnd = performance.now();
    console.log('training complete');
    console.log('training time:', Math.round(trainEnd - trainStart), 'milliseconds');
  });



  // do the testing

  const [images, labels] = getTestData();

  // save the images and labels in  global variables so we can
  // investigate them with console commands
  IMAGES_TO_EXAMINE = images;
  LABELS_TO_EXAMINE = labels;

  if (DO_TESTING) {
    for (let i = 0; i < NUM_IMAGES_TO_TEST; i++) {
      const testImage = images[i];
      const testLabel = labels[i];

      const testProbs = session.eval(outputLayer, [{tensor: inputLayer, data: testImage}]);
      const probs = testProbs.dataSync();
      const maxIndex = indexOfMax(probs);
      const topLabel = labelTag(maxIndex);
      const trueLabelIndex = indexOfMax(testLabel.dataSync());
      const trueLabel = labelTag(trueLabelIndex);
      const topProb = probs[maxIndex];

      console.log('*** prediction', i);
      console.log('true label:', trueLabel, 'predicted label:', topLabel, 'probability:', decimalToPercent(topProb));
      console.log('probabilities:', probs);
    }

  } // end of testing

}  //end of runModel


// run the model
// need to first noamalize the value to the range (-1, 1)
dataSet.fetchData().then(() => {
  dataSet.normalizeWithinBounds(0 /* 0 means normalize only images, not labels */, -1, 1);
  runModel();
});
