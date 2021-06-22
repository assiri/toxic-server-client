const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
const tfvis = require('@tensorflow/tfjs-vis');

const express = require('express');
const app = express();

app.get('/train', function (req, res) {
    console.log(tf.version);
    tf.ready().then(() => {
        const message = "Loaded TensorFlow.js - version: " + tf.version.tfjs + " \n with backend " + tf.getBackend();
        console.log(message);
        run();
        // training code 
        res.send(message);
    });
})

app.listen(9000, function (req, res) {
    console.log('Running server on port 9000 ...');
});


//const csvUrl = 'data/toxic_data_sample.csv';
//const csvUrl = "file://./src/assets/data/toxic_data_sample.csv";
app.use(express.static('src/assets'))
const csvUrl = "http://localhost:9000/data/toxic_data_sample.csv";


const stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
let tmpDictionary = {};
let EMBEDDING_SIZE = 1000;
const BATCH_SIZE = 16;
const render = false;
const TRAINING_EPOCHS = 10;

const readRawData = () => {
    // load data
    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    });
    return readData;
}

const plotOutputLabelCounts = (labels) => {
    const labelCounts = labels.reduce((acc, label) => {
        acc[label] = acc[label] === undefined ? 1 : acc[label] += 1;
        return acc;
    }, {});

    //console.log(labelCounts);
    const barChartData = [];
    Object.keys(labelCounts).forEach((key) => {
        barChartData.push({
            index: key,
            value: labelCounts[key]
        });
    });
    //console.log(barChartData);
    tfvis.render.barchart({
        tab: 'Exploration',
        name: 'Toxic output labels'
    }, barChartData);
}

const tokenize = (sentence, isCreateDict = false) => {
    const tmpTokens = sentence.split(/\s+/g);
    const tokens = tmpTokens.filter((token) => !stopwords.includes(token) && token.length > 0);

    if (isCreateDict) {
        const labelCounts = tokens.reduce((acc, token) => {
            acc[token] = acc[token] === undefined ? 1 : acc[token] += 1;
            return acc;
        }, tmpDictionary);
    }
    return tmpTokens;
}

const sortDictionaryByValue = (dict) => {
    const items = Object.keys(dict).map((key) => {
        return [key, dict[key]];
    });
    return items.sort((first, second) => {
        return second[1] - first[1];
    });
}

const getInverseDocumentFrequency = (documentTokens, dictionary) => {
    return dictionary.map((token) => 1 + Math.log(documentTokens.length / documentTokens.reduce((acc, curr) => curr.includes(token) ? acc + 1 : acc, 0)))
}

const encoder = (sentence, dictionary, idfs) => {
    const tokens = tokenize(sentence);
    const tfs = getTermFrequency(tokens, dictionary);
    const tfidfs = getTfIdf(tfs, idfs);
    return tfidfs;
}

const getTermFrequency = (tokens, dictionary) => {
    return dictionary.map((token) => tokens.reduce((acc, curr) => curr == token ? acc + 1 : acc, 0))
}

const getTfIdf = (tfs, idfs) => {
    return tfs.map((element, index) => element * idfs[index])
}


const prepareData = (dictionary, idfs) => {

    const preprocess = ({ xs, ys }) => {
        const comment = xs['comment_text'];
        const trimedComment = comment.toLowerCase().trim();
        const encoded = encoder(trimedComment, dictionary, idfs);

        return {
            xs: tf.tensor2d([encoded], [1, dictionary.length]),
            ys: tf.tensor2d([ys['toxic']], [1, 1])
        }

    }

    // load data
    const readData = tf.data.csv(csvUrl, {
        columnConfigs: {
            toxic: {
                isLabel: true
            }
        }
    })
        .map(preprocess);

    return readData;
}

const prepareDataUsingGenerator = (comments, labels, dictionary, idfs) => {
    function* getFeatures() {
        for (let i = 0; i < comments.length; i++) {
            // Generate one sample at a time.
            const encoded = encoder(comments[i], dictionary, idfs);
            yield tf.tensor2d([encoded], [1, dictionary.length]);
        }
    }
    function* getLabels() {
        for (let i = 0; i < labels.length; i++) {
            yield tf.tensor2d([labels[i]], [1, 1]);
        }
    }

    const xs = tf.data.generator(getFeatures);
    const ys = tf.data.generator(getLabels);
    const ds = tf.data.zip({ xs, ys });
    return ds;

}

const trainValTestSplit = (ds, nrows) => {
    // Train Test Split 
    const trainingValidationCount = Math.round(nrows * 0.7);
    const trainingCount = Math.round(nrows * 0.6);

    const SEED = 7687547;

    const trainingValidationData =
        ds
            .shuffle(nrows, SEED)
            .take(trainingValidationCount);

    const testDataset =
        ds
            .shuffle(nrows, SEED)
            .skip(trainingValidationCount)
            .batch(BATCH_SIZE);

    const trainingDataset =
        trainingValidationData
            .take(trainingCount)
            .batch(BATCH_SIZE);

    const validationDataset =
        trainingValidationData
            .skip(trainingCount)
            .batch(BATCH_SIZE);

    // return values
    return {
        trainingDataset,
        validationDataset,
        testDataset
    };
}


const buildModel = () => {
    const model = tf.sequential();
    model.add(tf.layers.dense({
        inputShape: [EMBEDDING_SIZE],
        activation: "relu",
        units: 5
    }));
    model.add(tf.layers.dense({
        activation: "sigmoid",
        units: 1
    }));
    model.compile({
        loss: "binaryCrossentropy",
        optimizer: tf.train.adam(0.06),
        metrics: ["accuracy"]
    });
    model.summary();
    return model;
}


const trainModel = async (model, trainingDataset, validationDataset) => {

    const history = [];
    const surface = { name: 'onEpochEnd Performance ', tab: 'Training' };

    const batchHistory = [];
    const batchSurface = { name: 'onBatchEnd Performance', tab: 'Training' };

    const messageCallback = new tf.CustomCallback(
        {
            onEpochEnd: async (epoch, logs) => {
                history.push(logs);
                console.log("Epoch :" + epoch + " loss: " + logs.loss);
                if (render){
                    tfvis.show.history(surface, history, ['loss', 'val_loss', 'acc', 'val_acc']);
                }
            },
            onBatchEnd: async (batch, logs) => {
                batchHistory.push(logs);
                if (render){
                    tfvis.show.history(batchSurface, batchHistory, ['loss', 'val_loss', 'acc', 'val_acc']);
                }
               
            }
        });
    // const earlyStoppingCallback = tf.callbacks.earlyStopping({
    //         monitor: 'val_acc',
    //         minDelta: 0.3,
    //         patience: 5,
    //         verbose: 1
    //     });
        
    const trainResult = await model.fitDataset(trainingDataset,
        {
            epochs: TRAINING_EPOCHS,
            validationData: validationDataset,
            callbacks: [
                messageCallback,
                //earlyStoppingCallback
            ]
            
        }
    );
    return model;

}

const evaluateModel = async (model, testDataset) => {
    const modeResult = await model.evaluateDataset(testDataset);

    const testLoss = modeResult[0].dataSync()[0];
    const testAcc = modeResult[1].dataSync()[0];
    console.log(`Loss on Test Dataset : ${testLoss.toFixed(4)}`);
    console.log(`Accuracy on Test Dataset : ${testAcc.toFixed(4)}`);

}


const getMoreEvaluationSummaries = async (model, testDataset) => {
    const allActualLables = [];
    const allPredictedLables = [];

    await testDataset.forEachAsync((row) => {
        //Actual labels
        const actualLabels = row['ys'].dataSync();
        actualLabels.forEach((x) => allActualLables.push(x));

         //Predicted labels
         const features = row['xs'];
         const predict  = model.predictOnBatch(tf.squeeze(features, 1)); 
         const predictLabels = tf.round(predict).dataSync();  // using 0.5 threshold
         predictLabels.forEach((x) => allPredictedLables.push(x));
    });

    // create actual and predicted label tensors
    const allActualLablesTensor = tf.tensor1d(allActualLables);
    const allPredictedLablesTensor = tf.tensor1d(allPredictedLables);
     
    // calculate accuracy result
    const accuracyResult = await tfvis.metrics.accuracy(allActualLablesTensor, allPredictedLablesTensor);
    console.log(`Accuracy result : ${accuracyResult}`);
     
    // calculate per class accuracy result
    const perClassAccuracyResult = await tfvis.metrics.perClassAccuracy(allActualLablesTensor, allPredictedLablesTensor);
    console.log(`Per Class Accuracy result : ${JSON.stringify(perClassAccuracyResult, null, 2)}`);

    // create confusion matrix report
    const confusionMatrixResult = await tfvis.metrics.confusionMatrix(allActualLablesTensor, allPredictedLablesTensor);
    const confusionMatrixVizResult = { "values": confusionMatrixResult };
    console.log(`confusion matrix : \n ${JSON.stringify(confusionMatrixVizResult, null,2)}`);
    const surface = { tab: 'Evaluation', name: 'Confusion Matrix' };
     if (render){
        tfvis.render.confusionMatrix(surface, confusionMatrixVizResult);
     }
}


const run = async () => {
    const rawDataResult = readRawData();
    const labels = [];

    const comments = [];
    const documentTokens = [];
    await rawDataResult.forEachAsync((row) => {
        //console.log(row);
        const comment = row['xs']['comment_text'];
        const trimedComment = comment.toLowerCase().trim();
        comments.push(trimedComment);
        documentTokens.push(tokenize(trimedComment, true));
        labels.push(row['ys']['toxic']);
    });

    // plot labels
    if(render){
        plotOutputLabelCounts(labels);
    }
    
    console.log(Object.keys(tmpDictionary).length);
    //console.log(tmpDictionary);

    const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
    if (sortedTmpDictionary.length <= EMBEDDING_SIZE) {
        EMBEDDING_SIZE = sortedTmpDictionary.length;
    }

    //console.log(sortedTmpDictionary);
    const dictionary = sortedTmpDictionary.slice(0, EMBEDDING_SIZE).map((row) => row[0]);

    // console.log('dictionary length ' + dictionary.length);
    // console.log(dictionary);

    // calculate IDF
    const idfs = getInverseDocumentFrequency(documentTokens, dictionary);
    //console.log(idfs);

    // Processing data 
    const ds = prepareData(dictionary, idfs);
    // await ds.forEachAsync((e) => console.log(e));

    //Sample test code
    // const documentTokens = [];
    // const testComments = ['i loved the movie', 'movie was boring'];
    // testComments.forEach((row) => {
    //     const comment = row.toLowerCase();
    //     documentTokens.push(tokenize(comment, true));
    // });

    // console.log(tmpDictionary);
    // const sortedTmpDictionary = sortDictionaryByValue(tmpDictionary);
    // const dictionary = sortedTmpDictionary.map((row) => row[0]);
    // const idfs = getInverseDocumentFrequency(documentTokens, dictionary);


    // testComments.forEach((row) => {
    //     const comment = row.toLowerCase();       
    //     console.log(encoder(comment,dictionary, idfs));
    // });

    // using generator
    // const ds = prepareDataUsingGenerator(comments, labels, dictionary, idfs)
    //await ds.forEachAsync((e) => console.log(e));

    const { trainingDataset, validationDataset, testDataset } = trainValTestSplit(ds, documentTokens.length);
    await trainingDataset.forEachAsync((e) => console.log(e));

    // Build Model 
    let model = buildModel();
    if (render) {
        tfvis.show.modelSummary({
            name: 'Model Summary',
            tab: 'Model'
        },
            model);
    }
    // Train Model 
    model = await trainModel(model, trainingDataset, validationDataset);

    // Evaluate Model 
    await evaluateModel(model, testDataset);
    await getMoreEvaluationSummaries(model, testDataset);

}