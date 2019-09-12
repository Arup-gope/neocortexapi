﻿using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Globalization;
using NeoCortexApi.Encoders;
using NeoCortexApi.Network;
using NeoCortexApi;
using NeoCortexApi.Entities;
using System.Diagnostics;

namespace UnitTestsProject
{
    [TestClass]
    public class NetworkTests
    {

        /// <summary>
        /// Initializes encoder and invokes Encode method.
        /// </summary>
        [TestMethod]
        [TestCategory("NetworkTests")]
        public void InitTests()
        {
            bool learn = true;
            Parameters p = Parameters.getAllDefaultParameters();
            p.Set(KEY.RANDOM, new ThreadSafeRandom(42));
            p.Set(KEY.INPUT_DIMENSIONS, new int[] { 100 });
            p.Set(KEY.CELLS_PER_COLUMN, 30);
            string[] categories = new string[] {"A", "B", "C", "D"};
            //string[] categories = new string[] { "A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L" , "M", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "Ö" };
            CortexNetwork net = new CortexNetwork("my cortex");
            List<CortexRegion> regions = new List<CortexRegion>();
            CortexRegion region0 = new CortexRegion("1st Region");
            regions.Add(region0);

            SpatialPoolerMT sp1 = new SpatialPoolerMT();
            TemporalMemory tm1 = new TemporalMemory();
            var mem = new Connections();
            p.apply(mem);
            sp1.init(mem, UnitTestHelpers.GetMemory());
            tm1.init(mem);
            Dictionary<string, object> settings = new Dictionary<string, object>();
            //settings.Add("W", 25);
            settings.Add("N", 100);
            //settings.Add("Radius", 1);

            EncoderBase encoder = new CategoryEncoder(categories, settings);
            //encoder.Encode()
            CortexLayer<object, object> layer1 = new CortexLayer<object, object>("L1");
            region0.AddLayer(layer1);
            layer1.HtmModules.Add(encoder);
            layer1.HtmModules.Add(sp1);
            //layer1.HtmModules.Add(tm1);
            //layer1.Compute();

            //IClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();
            HtmClassifier<string, ComputeCycle> cls = new HtmClassifier<string, ComputeCycle>();
            HtmClassifier_Test<string, ComputeCycle> cls1 = new HtmClassifier_Test<string, ComputeCycle>();
            //string[] inputs = new string[] { "A", "B", "C", "D" };
            string[] inputs = new string[] { "A", "B", "C", "D" };

            //
            // This trains SP.
            foreach (var input in inputs)
            {
                Debug.WriteLine($" ** {input} **");
                for (int i = 0; i < 3; i++)
                {
                    var lyrOut = layer1.Compute((object)input, learn) as ComputeCycle;
                }
            }

            layer1.HtmModules.Add(tm1);

            for (int i = 0; i < 200; i++)
            {
                foreach (var input in inputs)
                {
                    var lyrOut = layer1.Compute((object)input, learn) as ComputeCycle;
                    //cls1.Learn(input, lyrOut.activeCells.ToArray(), learn);
                    //Debug.WriteLine($"Current Input: {input}");
                    cls.Learn(input, lyrOut.activeCells.ToArray(), lyrOut.predictiveCells.ToArray());
                    Debug.WriteLine($"Current Input: {input}");
                    if (learn == false)
                    {
                        Debug.WriteLine($"Predict Input When Not Learn: {cls.GetPredictedInputValue(lyrOut.predictiveCells.ToArray())}");
                    }
                    else
                    {
                        Debug.WriteLine($"Predict Input: {cls.GetPredictedInputValue(lyrOut.predictiveCells.ToArray())}");
                    }
                    
                    //Debug.WriteLine("-----------------------------------------------------------\n----------------------------------------------------------");
                }

                tm1.reset(mem);
                if (i == 20)
                {
                    Debug.WriteLine("Stop Learning From Here-----------------------------------------------------------------------------------------------------\n"
                        +"-----------------------------------------------------------------------------------------------" +
                        "-------------------------------------------------------------------------------------------------");
                    learn = false;
                    //tm1.reset(mem);
                    
                }

                tm1.reset(mem);
            }

            Debug.WriteLine("------------------------------------------------------------------------\n----------------------------------------------------------------------------");
            /*
            learn = false;
            for (int i = 0; i < 19; i++)
            {
                foreach (var input in inputs)
                {
                    layer1.Compute((object)input, learn);
                }
            }
            */

        }

    }
}
