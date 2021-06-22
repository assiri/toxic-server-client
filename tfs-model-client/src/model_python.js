import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import dictionary from './assets/models/tfjs_python_toxicity/dictionary.json'
import idfs from './assets/models/tfjs_python_toxicity/idfs.json'



export const load = () => {
    return loadModel();
};

export const predict = (sentence, model) => {
    console.log("predict using custom model");
    return predictResults(sentence, model);
};

const MODEL_PATH = 'models/tfjs_python_toxicity/model.json';


const stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']
let tmpDictionary = {};



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










const loadModel = async () => {
    const model_python = await tf.loadLayersModel(MODEL_PATH);
    return model_python;
    // const modelPath = `localstorage://${MODEL_ID}`;
    // console.log(modelPath);
    // // Load Model 
    // const models = await tf.io.listModels();
    // if (models[modelPath]) {
    //     // load Model 
    //     console.log('model exists');
    //     const model_loaded = await tf.loadLayersModel(modelPath);
    //     return model_loaded;
    // }
    // else {
    //     console.log('no model avaialble');
    //     return null;
    // }

}

const predictResults = async function (sentence, model) {
    console.log('Prediction started');
    // // load IDFs
    // const idfObject = localStorage.getItem(IDF_STORAGE_ID);
    // const idfs = JSON.parse(idfObject);

    // // load Dictionary
    // const dictionaryObject = localStorage.getItem(DICTIONARY_STORAGE_ID);
    // const dictionary = JSON.parse(dictionaryObject);

    // get encoded values
    const encoded = encoder(sentence.toLowerCase().trim(), dictionary, idfs);
    const encodedTensor = tf.tensor2d([encoded], [1, dictionary.length]);

    // make predictions
    const predictionResult = model.predict(encodedTensor);
    const predictionScore = predictionResult.dataSync();

    // extract prdiction class
    const predictedClass = (predictionScore >= 0.5) ? "toxic" : "non-toxic";
    const resultMessage = `Probability : ${predictionScore}, Class : ${predictedClass}`;
    console.log(resultMessage);
    return predictedClass;




}



