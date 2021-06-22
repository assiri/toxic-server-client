import * as tf from '@tensorflow/tfjs';
import * as toxicity from '@tensorflow-models/toxicity';

export const load = () => {
    return loadModel();
};
export const predict = (sentence, model) => {
    console.log("predict using Toxicity model");
    return predictResults(sentence, model);
};


const threshold = 0.9; // default 0.85, Prediction probabilities

const loadModel = async function () {
    const model = await toxicity.load(threshold);
    return model;

}
const predictResults = async function (sentence, model) {
    console.log('Prediction started');
   
    // make predictions
    const predictionResult = await model.classify(sentence.toLowerCase().trim());
    console.log(predictionResult);

    // extract predicted classes
    let predictedClasses = [];
    predictionResult.forEach((e) => {
        if (e['results'][0]['match'] == true){
            predictedClasses.push(e['label']);
        }
    });
    if (predictedClasses.length <= 0){
        predictedClasses.push("non-toxic");
    }
    console.log(predictedClasses);
    return predictedClasses;
}