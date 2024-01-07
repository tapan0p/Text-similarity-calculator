import { HfInference } from '@huggingface/inference';
const hf = new HfInference('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');
function dotProduct(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        result += a[i] * b[i];
    }
    return result;
}
async function main() {
    return await hf.featureExtraction({
        model: "intfloat/multilingual-e5-small",
        inputs: "IIT Guwahati is the best iit",
    });
}
async function run() {
    const output1 = await main();
    const output2 = await main();
    let similarityscore = dotProduct(output1, output2);
    console.log(similarityscore);
}
run();
