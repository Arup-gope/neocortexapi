// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi.Classifiers;
using NeoCortexApi.Encoders;
using NeoCortexApi.Entities;
using NeoCortexApi.Network;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace NeoCortexApi.Experiments
{
    /// <summary>
    /// Check out student paper in the following URL: https://github.com/ddobric/neocortexapi/blob/master/NeoCortexApi/Documentation/Experiments/ML-19-20_20-5.4_HtmSparsityExperiments_Paper.pdf
    /// </summary>
    [TestClass]
    public class HtmClassifierTest
    {
        private void createHtmClassifier(CortexLayer<object, object> layer1,
            HtmClassifier<string, ComputeCycle> cls,
            Dictionary<string, List<double>> sequences)
        {
            int inputBits = 100;
            int numColumns = 1024;

            HtmConfig cfg = new HtmConfig(new int[] { inputBits }, new int[] { numColumns })
            {
                Random = new ThreadSafeRandom(42),

                CellsPerColumn = 25,
                GlobalInhibition = true,
                LocalAreaDensity = -1,
                NumActiveColumnsPerInhArea = 0.02 * numColumns,
                PotentialRadius = (int)(0.15 * inputBits),
                //InhibitionRadius = 15,

                MaxBoost = 10.0,
                DutyCyclePeriod = 25,
                MinPctOverlapDutyCycles = 0.75,
                MaxSynapsesPerSegment = (int)(0.02 * numColumns),

                ActivationThreshold = 15,
                ConnectedPermanence = 0.5,

                // Learning is slower than forgetting in this case.
                PermanenceDecrement = 0.25,
                PermanenceIncrement = 0.15,

                // Used by punishing of segments.
                PredictedSegmentDecrement = 0.1
            };

            double max = 20;

            Dictionary<string, object> settings = new Dictionary<string, object>()
            {
                { "W", 15},
                { "N", inputBits},
                { "Radius", -1.0},
                { "MinVal", 0.0},
                { "Periodic", false},
                { "Name", "scalar"},
                { "ClipInput", false},
                { "MaxVal", max}
            };

            int maxMatchCnt = 0;

            EncoderBase encoder = new ScalarEncoder(settings);

            var mem = new Connections(cfg);

            //HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();

            //CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");

            TemporalMemory tm = new TemporalMemory();

            SpatialPoolerMT sp = new SpatialPoolerMT();
            sp.Init(mem);
            tm.Init(mem);
            layer1.HtmModules.Add("encoder", encoder);
            layer1.HtmModules.Add("sp", sp);

            //double[] inputs = inputValues.ToArray();
            int[] prevActiveCols = new int[0];

            int cycle = 0;
            int matches = 0;

            var lastPredictedValues = new List<string>(new string[] { "0" });

            int maxCycles = 100;

            for (int i = 0; i < maxCycles; i++)
            {
                matches = 0;

                cycle++;


                foreach (var inputs in sequences)
                {
                    foreach (var input in inputs.Value)
                    {

                        var lyrOut = layer1.Compute(input, true);
                    }
                }
            }

            layer1.HtmModules.Add("tm", tm);

            foreach (var sequenceKeyPair in sequences)
            {

                int maxPrevInputs = sequenceKeyPair.Value.Count - 1;

                List<string> previousInputs = new List<string>();

                previousInputs.Add("-1.0");

                //
                // Now training with SP+TM. SP is pretrained on the given input pattern set.
                for (int i = 0; i < maxCycles; i++)
                {
                    matches = 0;

                    cycle++;

                    foreach (var input in sequenceKeyPair.Value)
                    {

                        var lyrOut = layer1.Compute(input, true) as ComputeCycle;

                        var activeColumns = layer1.GetResult("sp") as int[];

                        previousInputs.Add(input.ToString());
                        if (previousInputs.Count > (maxPrevInputs + 1))
                            previousInputs.RemoveAt(0);

                        // In the pretrained SP with HPC, the TM will quickly learn cells for patterns
                        // In that case the starting sequence 4-5-6 might have the sam SDR as 1-2-3-4-5-6,
                        // Which will result in returning of 4-5-6 instead of 1-2-3-4-5-6.
                        // HtmClassifier allways return the first matching sequence. Because 4-5-6 will be as first
                        // memorized, it will match as the first one.
                        if (previousInputs.Count < maxPrevInputs)
                            continue;

                        string key = GetKey(previousInputs, input, sequenceKeyPair.Key);

                        List<Cell> actCells;

                        if (lyrOut.ActiveCells.Count == lyrOut.WinnerCells.Count)
                        {
                            actCells = lyrOut.ActiveCells;
                        }
                        else
                        {
                            actCells = lyrOut.WinnerCells;
                        }

                        cls.Learn(key, actCells.ToArray());

                        //
                        // If the list of predicted values from the previous step contains the currently presenting value,
                        // we have a match.
                        if (lastPredictedValues.Contains(key))
                        {
                            matches++;
                        }

                        if (lyrOut.PredictiveCells.Count > 0)
                        {
                            //var predictedInputValue = cls.GetPredictedInputValue(lyrOut.PredictiveCells.ToArray());
                            var predictedInputValues = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 3);

                            lastPredictedValues = predictedInputValues.Select(v => v.PredictedInput).ToList();
                        }
                        else
                        {
                            lastPredictedValues = new List<string>();
                        }
                    }

                    // The first element (a single element) in the sequence cannot be predicted
                    double maxPossibleAccuraccy = (double)((double)sequenceKeyPair.Value.Count - 1) / (double)sequenceKeyPair.Value.Count * 100.0;

                    double accuracy = (double)matches / (double)sequenceKeyPair.Value.Count * 100.0;

                    if (accuracy >= maxPossibleAccuraccy)
                    {
                        maxMatchCnt++;
                        //
                        // Experiment is completed if we are 30 cycles long at the 100% accuracy.
                        if (maxMatchCnt >= 30)
                        {
                            break;
                        }
                    }
                    else if (maxMatchCnt > 0)
                    {
                        maxMatchCnt = 0;
                    }

                    // This resets the learned state, so the first element starts allways from the beginning.
                    tm.Reset(mem);
                }
            }
        }

        private static string GetKey(List<string> prevInputs, double input, string sequence)
        {
            string key = String.Empty;

            for (int i = 0; i < prevInputs.Count; i++)
            {
                if (i > 0)
                    key += "-";

                key += (prevInputs[i]);
            }

            return $"{sequence}_{key}";
        }
        /// <summary>
        ///Htm Sparsity is the ratio between Width and InputBits(W/N). This unit test runs in a loop and saves cycle at which
        ///we get 100% match for the first time at Sparsity=0.18. W and N can be changed but the ratio must be 0.18.
        ///This program has 2 loops (loop inside a loop), the parent loop/outer loop is defined keeping in mind how many
        ///readings are wanted in the result. The child loop/inner loop has 460 cycle, but is ended as soon as we get 100%
        ///match i.e. for max=10 10 out 10 matches. Then the parent loop is incremented and it continues for the number of
        ///loops defined (in our case we used 1000 - 10000 loops).
        ///"We found out that, for max=10 ideal HTM Sparsity is 0.18"
        /// </summary>

        [TestMethod]
        public void CheckNextInputValue()
        {
            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();

            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");

            Dictionary<string, List<double>> sequences = new Dictionary<string, List<double>>();

            sequences.Add("S1", new List<double>(new double[] { 0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 5.0, }));

            createHtmClassifier(layer1,cls,sequences);

            var list = new double[] { 1.0, 2.0, 3.0 };
            var predictedVal = new double[] { 2.0, 3.0, 4.0 };
            var index = 0;
            foreach (var item in list)
            {
                var lyrOut = layer1.Compute(item, false) as ComputeCycle;

                var res = cls.GetPredictedInputValues(lyrOut.PredictiveCells.ToArray(), 1);

                var tokens = res.First().PredictedInput.Split('_');
                var tokens2 = res.First().PredictedInput.Split('-');
                var predictValue = Convert.ToInt32(tokens2[tokens.Length - 1]);
            }
        }
    }
}