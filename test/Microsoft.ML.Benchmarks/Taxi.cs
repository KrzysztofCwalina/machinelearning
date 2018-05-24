// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;

namespace Microsoft.ML.Benchmarks
{
    public class TaxiFareBench
    {
        static string s_metric;
        private static PredictionModel<TaxiFareData, TaxiFarePrediction> s_trainedModel;
        private static string s_dataPath;
        private static TaxiFareData[][] s_batches;
        private static readonly int[] s_batchSizes = new int[] { 1, 2, 5 };
        private readonly Random r = new Random(0);
        private readonly static TaxiFareData s_example = new TaxiFareData()
        {
            PassengerCount = 1,
            RateCode = "1",
            TripDistance = 1000,
            TripTime = 30,
            VendorId = "VTS",
            PaymentType = "CRD"
        };


        [Benchmark]
        public PredictionModel<TaxiFareData, TaxiFarePrediction> Train() => TrainCore();

        //[Benchmark]
        //public float Predict() => s_trainedModel.Predict(s_example).FareAmount;

        //[Benchmark]
        //public IEnumerable<TaxiFarePrediction> PredictBatchOf1() => s_trainedModel.Predict(s_batches[0]);
        //[Benchmark]
        //public IEnumerable<TaxiFarePrediction> PredictBatchOf2() => s_trainedModel.Predict(s_batches[1]);
        //[Benchmark]
        //public IEnumerable<TaxiFarePrediction> PredictBatchOf5() => s_trainedModel.Predict(s_batches[2]);

        [GlobalSetup]
        public void Setup()
        {
            s_dataPath = Program.GetDataPath("taxi-fare-test.csv");
            s_trainedModel = TrainCore();
            TaxiFarePrediction prediction = s_trainedModel.Predict(s_example);

            var testData = new TextLoader(s_dataPath).CreateFrom<TaxiFareData>(useHeader: true);
            var evaluator = new RegressionEvaluator();
            var metrics = evaluator.Evaluate(s_trainedModel, testData);
            s_metric = metrics.Rms.ToString();

            s_batches = new TaxiFareData[s_batchSizes.Length][];
            for (int i = 0; i < s_batches.Length; i++)
            {
                var batch = new TaxiFareData[s_batchSizes[i]];
                s_batches[i] = batch;
                for (int bi = 0; bi < batch.Length; bi++)
                {
                    batch[bi] = s_example;
                }
            }
        }

        private static PredictionModel<TaxiFareData, TaxiFarePrediction> TrainCore()
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(s_dataPath).CreateFrom<TaxiFareData>(useHeader: true, separator: ','));
            pipeline.Add(new ColumnCopier(("FareAmount", "Label")));
            pipeline.Add(new CategoricalOneHotVectorizer("VendorId",
                    "RateCode",
                    "PaymentType"));
            pipeline.Add(new ColumnConcatenator("Features",
                    "VendorId",
                    "RateCode",
                    "PassengerCount",
                    "TripDistance",
                    "PaymentType"));
            pipeline.Add(new FastTreeRegressor());

            PredictionModel<TaxiFareData, TaxiFarePrediction> model = pipeline.Train<TaxiFareData, TaxiFarePrediction>();
            return model;
        }

        public class TaxiFareData
        {
            [Column("0")] public string VendorId;

            [Column("1")] public string RateCode;

            [Column("2")] public float PassengerCount;

            [Column("3")] public float TripTime;

            [Column("4")] public float TripDistance;

            [Column("5")] public string PaymentType;

            [Column("6")] public float FareAmount;         
        }

        public class TaxiFarePrediction
        {
            [ColumnName("Score")]
            public float FareAmount;
        }
    }
}
