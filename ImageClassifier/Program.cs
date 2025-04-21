using ImageClassifier;
using Microsoft.ML;
using Microsoft.ML.Data;

string _assetsPath = Path.Combine(Environment.CurrentDirectory, "assets");
string _imagesFolder = Path.Combine(_assetsPath, "images");
string _trainTagsTsv = Path.Combine(_imagesFolder, "tags.tsv");
string _testTagsTsv = Path.Combine(_imagesFolder, "test-tags.tsv");
//string _predictSingleImage = Path.Combine(_imagesFolder, "toaster3.jpg");
//string _predictSingleImage = Path.Combine(_imagesFolder, "weird-t.jpg");
//string _predictSingleImage = Path.Combine(_imagesFolder, "face.jpg");
string _predictSingleImage = Path.Combine(_imagesFolder, "toa.jpg");
string _inceptionTensorFlowModel = Path.Combine(_assetsPath, "inception", "tensorflow_inception_graph.pb");

MLContext mlContext = new MLContext();
ITransformer model = GenerateModel(mlContext);
ClassifySingleImage(mlContext, model);


void ClassifySingleImage(MLContext mlContext, ITransformer model)
{
    var imageData = new ImageData()
    {
        ImagePath = _predictSingleImage
    };

    // Make prediction function (input = ImageData, output = ImagePrediction)
    var predictor = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(model);

    //note: for improved performance and thread safety in production environments, use the PredictionEnginePool service,
    //which creates an ObjectPool of PredictionEngine objects for use throughout your application.
    //See this guide on how to use PredictionEnginePool in an ASP.NET Core Web API.
    var prediction = predictor.Predict(imageData);

    Console.WriteLine($"Classifying single image. Image: {Path.GetFileName(imageData.ImagePath)} predicted as: {prediction.PredictedLabelValue} " +
        $"with score: {prediction.Score?.Max()} ");
}

ITransformer GenerateModel(MLContext mlContext)
{
    IEstimator<ITransformer> pipeline = mlContext.Transforms.LoadImages(outputColumnName: "input", imageFolder: _imagesFolder, 
        inputColumnName: nameof(ImageData.ImagePath))
                // The image transforms transform the images into the model's expected format.
                .Append(mlContext.Transforms.ResizeImages(outputColumnName: "input", imageWidth: InceptionSettings.ImageWidth, 
                imageHeight: InceptionSettings.ImageHeight, inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", 
                interleavePixelColors: InceptionSettings.ChannelsLast, offsetImage: InceptionSettings.Mean))
                .Append(mlContext.Model.LoadTensorFlowModel(_inceptionTensorFlowModel).
                ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2_pre_activation" }, inputColumnNames: new[] { "input" }, 
                addBatchDimensionInput: true))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "LabelKey", inputColumnName: "Label"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: "LabelKey", 
                featureColumnName: "softmax2_pre_activation"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabelValue", "PredictedLabel"))
                .AppendCacheCheckpoint(mlContext);

    IDataView trainingData = mlContext.Data.LoadFromTextFile<ImageData>(path: _trainTagsTsv, hasHeader: false);

    ITransformer model = pipeline.Fit(trainingData);

    IDataView testData = mlContext.Data.LoadFromTextFile<ImageData>(path: _testTagsTsv, hasHeader: false);
    IDataView predictions = model.Transform(testData);

    // Create an IEnumerable for the predictions for displaying results
    IEnumerable<ImagePrediction> imagePredictionData = mlContext.Data.CreateEnumerable<ImagePrediction>(predictions, true);
    DisplayResults(imagePredictionData);

    MulticlassClassificationMetrics metrics =
    mlContext.MulticlassClassification.Evaluate(predictions,
        labelColumnName: "LabelKey",
        predictedLabelColumnName: "PredictedLabel");

    Console.WriteLine($"LogLoss is: {metrics.LogLoss}");
    Console.WriteLine($"PerClassLogLoss is: {String.Join(" , ", metrics.PerClassLogLoss.Select(c => c.ToString()))}");

    return model;
}

void DisplayResults(IEnumerable<ImagePrediction> imagePredictionData)
{
    foreach (ImagePrediction prediction in imagePredictionData)
    {
        Console.WriteLine($"From image prediction data. Image: {Path.GetFileName(prediction.ImagePath)} predicted as: {prediction.PredictedLabelValue} " +
            $"with score: {prediction.Score?.Max()} ");
    }
}
