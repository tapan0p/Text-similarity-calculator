import { pipeline } from '@xenova/transformers';
let generateEmbaddings = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
let output1 = await generateEmbaddings('John is a happy man.', { pooling: 'mean', normalize: true });
let output2 = await generateEmbaddings('John is a happy male.', { pooling: 'mean', normalize: true });
function dotProduct(a, b) {
    let result = 0;
    for (let i = 0; i < a.length; i++) {
        result += a[i] * b[i];
    }
    return result;
}
let similarityScore = dotProduct(output1.data, output2.data);
console.log(similarityScore);
