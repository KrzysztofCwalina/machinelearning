// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using Microsoft.ML.Data;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Xunit;

namespace Microsoft.ML.Scenarios
{
    public partial class ScenariosTests
    {
        [Fact]
        public void TrainAndPredictTaxiFare()
        {
            string dataPath = GetDataPath("taxi-fare-train.csv");
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(dataPath).CreateFrom<TaxiFareData>(useHeader: true, separator: ','));
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

            TaxiFareData example = new TaxiFareData()
            {
                PassengerCount = 1,
                RateCode = "1",
                TripDistance = 1000,
                TripTime = 30,
                VendorId = "VTS",
                PaymentType = "CRD"
            };

            TaxiFarePrediction prediction = model.Predict(example);
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
