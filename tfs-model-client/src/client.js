import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import '@tensorflow/tfjs-backend-wasm';
import "regenerator-runtime/runtime";
import * as model from './model';
import * as modelPython from './model_python';
import * as modelTransfer from './model_transfer';
import * as modelToxicity from './model_toxicity';

import $ from "jquery";
import "materialize-css";
import "material-icons";
import "./main.scss";

M.AutoInit();




const init = async () => {
    await tf.ready();
    const init_message = "Powered by TensorFlow.js - version: " + tf.version.tfjs + " with backend : " + tf.getBackend();
    $('#init').text(init_message);
    // console.log(tf.getBackend());
    // model.train();
}
init();


var modelOptionSelect = $('select');
let modelOption = 1;
modelOptionSelect.on('change',  (e) => {
    modelOption = parseInt(e.target.value);
});

$("#toggle_visor").on('change',  () => {
    tfvis.visor().toggle();
});


$('#btn_train').on('click', async () => {

    switch (modelOption) {
        case 1:
            $('#btn_train').addClass("disabled");
            M.toast({ html: 'Training Started!' })
            console.log("Training custom model with TFIDF features");
            await model.train();
            break;
        case 2:
            M.toast({ html: 'No training needed for Python exported model!' })
            break;
        case 3:
            $('#btn_train').addClass("disabled");
            M.toast({ html: 'Training Started!' })
            console.log("Transfer learning on pre-trained USE Model");
            await modelTransfer.train();
            break;
        case 4:
            M.toast({ html: 'No training needed Toxicity pre-trained model!' })
            break;
    
        default:
            break;
    }

});
let model_loaded;
$('#btn_load').on('click', async function () {

    switch (modelOption) {
        case 1:
            console.log("Loading TF.js trained model");
            model_loaded = await model.load();
            model_loaded.summary();
            break;

        case 2:
            console.log("Loading python exported model");
            model_loaded = await modelPython.load();
            model_loaded.summary();
            break;
        case 3:
            console.log("Loading pre-trained transfer learning exported model");
            model_loaded = await modelTransfer.load();
            model_loaded.summary();
            break;
        case 4:
            console.log("Loading pre-trained Toxicity model");
            model_loaded = await modelToxicity.load();
            console.log("Loaded.");
            break;
        default:
            break;
    }

    $('#btn_load').addClass("disabled");
    $('#btn_predict').removeClass("disabled");
});

$('#btn_predict').on('click', async function () {
    const message = $('#textarea-message').val();
    if (message.trim().length <= 0){
        M.toast({ html: 'Empty message.Nothing to predict!' });
        return;
    }
    console.log("Here is the message : " + message);

    $("#chip_result").empty();
    $('#btn_predict').addClass("disabled");
    let predictedClasses = null;
    switch (modelOption) {
        case 1:
            predictedClasses = await model.predict(message, model_loaded);
            $("#chip_result").append(`Predicted Label : <div class="chip pink-text">${predictedClasses}</div>`);
            break;
        case 2:
            predictedClasses = await modelPython.predict(message, model_loaded);
            $("#chip_result").append(`Predicted Label : <div class="chip pink-text">${predictedClasses}</div>`);
            break;
        case 3:
            predictedClasses = await modelTransfer.predict(message, model_loaded);
            $("#chip_result").append(`Predicted Label : <div class="chip pink-text">${predictedClasses}</div>`);
            break;
        case 4:
            predictedClasses = await modelToxicity.predict(message, model_loaded);
            $('#chip_result').append('<span>Predicted Label(s): </span');
            predictedClasses.forEach((element) => {
                $("#chip_result").append(`<div class="chip pink-text">${element}</div>`);
            });
            break;
                
        default:
            break;
    }

    $('#btn_predict').removeClass("disabled");

});