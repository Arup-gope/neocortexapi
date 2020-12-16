﻿// Copyright (c) Damir Dobric. All rights reserved.
// Licensed under the Apache License, Version 2.0. See LICENSE in the project root for license information.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeoCortexApi;
using NeoCortexApi.Entities;
using NeoCortexApi.Utility;
using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.IO;
using System.Runtime.Serialization;
using NeoCortexApi.Types;
using Newtonsoft.Json;
using Newtonsoft.Json.Serialization;

namespace UnitTestsProject
{
    [TestClass]
    public class TemporalPoolerTest
    {
        private static bool areDisjoined<T>(ICollection<T> arr1, ICollection<T> arr2)
        {
            foreach (var item in arr1)
            {
                if (arr2.Contains(item))
                    return false;
            }

            return true;
        }
        //private Parameters getDefaultParameters()
        //{
        //    Parameters retVal = Parameters.getTemporalDefaultParameters();
        //    retVal.Set(KEY.COLUMN_DIMENSIONS, new int[] { 32 });
        //    retVal.Set(KEY.CELLS_PER_COLUMN, 4);
        //    retVal.Set(KEY.ACTIVATION_THRESHOLD, 3);
        //    retVal.Set(KEY.INITIAL_PERMANENCE, 0.21);
        //    retVal.Set(KEY.CONNECTED_PERMANENCE, 0.5);
        //    retVal.Set(KEY.MIN_THRESHOLD, 2);
        //    retVal.Set(KEY.MAX_NEW_SYNAPSE_COUNT, 3);
        //    retVal.Set(KEY.PERMANENCE_INCREMENT, 0.10);
        //    retVal.Set(KEY.PERMANENCE_DECREMENT, 0.10);
        //    retVal.Set(KEY.PREDICTED_SEGMENT_DECREMENT, 0.0);
        //    retVal.Set(KEY.RANDOM, new ThreadSafeRandom(42));
        //    retVal.Set(KEY.SEED, 42);

        //    return retVal;
        //}

        private HtmConfig GetDefaultTMParameters()
        {
            HtmConfig htmConfig = new HtmConfig()
            {
                ColumnDimensions = new int[] { 32 },
                CellsPerColumn = 4,
                ActivationThreshold = 3,
                InitialPermanence = 0.21,
                ConnectedPermanence = 0.5,
                MinThreshold = 2,
                MaxNewSynapseCount = 3,
                PermanenceIncrement = 0.1,
                PermanenceDecrement = 0.1,
                PredictedSegmentDecrement = 0,
                Random = new ThreadSafeRandom(42),
                RandomGenSeed = 42
            };

            return htmConfig;
        }


        //private Parameters getDefaultParameters(Parameters p, string key, Object value)
        //{
        //    Parameters retVal = p == null ? getDefaultParameters() : p;
        //    retVal.Set(key, value);

        //    return retVal;
        //}


        private T deepCopyPlain<T>(T obj)
        {
            IFormatter formatter = new BinaryFormatter();
            using (Stream stream = new MemoryStream())
            {
                formatter.Serialize(stream, obj);
                stream.Position = 0;
                return (T)formatter.Deserialize(stream);
            }

            //JsonSerializerSettings jss = new Newtonsoft.Json.JsonSerializerSettings();

            //string serObj = JsonConvert.SerializeObject(obj);
            //return JsonConvert.DeserializeObject<T>(serObj);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestActivateCorrectlyPredictiveCells()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            int[] activeColumns = { 1 };

            // Cell4 belongs to column with index 1.
            Cell cell4 = cn.GetCell(4);

            // ISet<Cell> expectedActiveCells = Stream.of(cell4).collect(Collectors.toSet());
            ISet<Cell> expectedActiveCells = new HashSet<Cell>(new Cell[] { cell4 });

            // We add distal dentrite at column1.cell4
            DistalDendrite activeSegment = cn.CreateDistalSegment(cell4);

            //
            // We add here synapses between column0.cells[0-3] and segment.
            cn.CreateSynapse(activeSegment, cn.GetCell(0), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(1), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(2), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(3), 0.5);

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            Assert.IsTrue(cc.PredictiveCells.SequenceEqual(expectedActiveCells));

            ComputeCycle cc2 = tm.Compute(activeColumns, true) as ComputeCycle;
            Assert.IsTrue(cc2.ActiveCells.SequenceEqual(expectedActiveCells));
        }

        [TestMethod]
        public void TestBurstUnpredictedColumns()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] activeColumns = { 0 };
            IList<Cell> burstingCells = cn.GetCellSet(new int[] { 0, 1, 2, 3 });

            ComputeCycle cc = tm.Compute(activeColumns, true) as ComputeCycle;

            Assert.IsTrue(cc.ActiveCells.SequenceEqual(burstingCells));
        }

        [TestMethod]
        public void TestBurstUnpredictedColumns1()
        {
            HtmConfig htmConfig = GetDefaultTMParameters();
            Connections cn = new Connections(htmConfig);

            TemporalMemory tm = new TemporalMemory();

            tm.Init(cn);

            int[] activeColumns = { 0 };
            IList<Cell> burstingCells = cn.GetCellSet(new int[] { 0, 1, 2, 3 });

            ComputeCycle cc = tm.Compute(activeColumns, true) as ComputeCycle;

            Assert.IsTrue(cc.ActiveCells.SequenceEqual(burstingCells));
        }


        [TestMethod]
        [TestCategory("Prod")]
        public void TestZeroActiveColumns()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            Cell cell4 = cn.GetCell(4);

            DistalDendrite activeSegment = cn.CreateDistalSegment(cell4);
            cn.CreateSynapse(activeSegment, cn.GetCell(0), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(1), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(2), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(3), 0.5);

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            Assert.IsFalse(cc.ActiveCells.Count == 0);
            Assert.IsFalse(cc.WinnerCells.Count == 0);
            Assert.IsFalse(cc.PredictiveCells.Count == 0);

            int[] zeroColumns = new int[0];
            ComputeCycle cc2 = tm.Compute(zeroColumns, true) as ComputeCycle;
            Assert.IsTrue(cc2.ActiveCells.Count == 0);
            Assert.IsTrue(cc2.WinnerCells.Count == 0);
            Assert.IsTrue(cc2.PredictiveCells.Count == 0);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestPredictedActiveCellsAreAlwaysWinners()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            int[] activeColumns = { 1 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            List<Cell> expectedWinnerCells = new List<Cell>(cn.GetCellSet(new int[] { 4, 6 }));

            DistalDendrite activeSegment1 = cn.CreateDistalSegment(expectedWinnerCells[0]);
            cn.CreateSynapse(activeSegment1, previousActiveCells[0], 0.5);
            cn.CreateSynapse(activeSegment1, previousActiveCells[1], 0.5);
            cn.CreateSynapse(activeSegment1, previousActiveCells[2], 0.5);

            DistalDendrite activeSegment2 = cn.CreateDistalSegment(expectedWinnerCells[1]);
            cn.CreateSynapse(activeSegment2, previousActiveCells[0], 0.5);
            cn.CreateSynapse(activeSegment2, previousActiveCells[1], 0.5);
            cn.CreateSynapse(activeSegment2, previousActiveCells[2], 0.5);

            ComputeCycle cc = tm.Compute(previousActiveColumns, false) as ComputeCycle; // learn=false
            cc = tm.Compute(activeColumns, false) as ComputeCycle; // learn=false

            Assert.IsTrue(cc.WinnerCells.SequenceEqual(new LinkedHashSet<Cell>(expectedWinnerCells)));
        }


        [TestMethod]
        public void TestReinforcedCorrectlyActiveSegments()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.InitialPermanence = 0.2;
            htmConfig.PermanenceDecrement = 0.08;
            htmConfig.PredictedSegmentDecrement = 0.02;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            int[] activeColumns = { 1 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            Cell activeCell = cn.GetCell(5);

            DistalDendrite activeSegment = cn.CreateDistalSegment(activeCell);
            Synapse as1 = cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.5);
            Synapse as2 = cn.CreateSynapse(activeSegment, previousActiveCells[1], 0.5);
            Synapse as3 = cn.CreateSynapse(activeSegment, previousActiveCells[2], 0.5);
            Synapse is1 = cn.CreateSynapse(activeSegment, cn.GetCell(81), 0.5);

            tm.Compute(previousActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0.6, as1.Permanence, 0.1);
            Assert.AreEqual(0.6, as2.Permanence, 0.1);
            Assert.AreEqual(0.6, as3.Permanence, 0.1);
            Assert.AreEqual(0.42, is1.Permanence, 0.001);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestReinforcedSelectedMatchingSegmentInBurstingColumn()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.PermanenceDecrement = 0.08;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            int[] activeColumns = { 1 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            Cell[] burstingCells = { cn.GetCell(4), cn.GetCell(5) };

            DistalDendrite activeSegment = cn.CreateDistalSegment(burstingCells[0]);
            Synapse as1 = cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.3);
            Synapse as2 = cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.3);
            Synapse as3 = cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.3);
            Synapse is1 = cn.CreateSynapse(activeSegment, cn.GetCell(81), 0.3);

            DistalDendrite otherMatchingSegment = cn.CreateDistalSegment(burstingCells[1]);
            cn.CreateSynapse(otherMatchingSegment, previousActiveCells[0], 0.3);
            cn.CreateSynapse(otherMatchingSegment, previousActiveCells[1], 0.3);
            cn.CreateSynapse(otherMatchingSegment, cn.GetCell(81), 0.3);

            tm.Compute(previousActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0.4, as1.Permanence, 0.01);
            Assert.AreEqual(0.4, as2.Permanence, 0.01);
            Assert.AreEqual(0.4, as3.Permanence, 0.01);
            Assert.AreEqual(0.22, is1.Permanence, 0.001);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestNoChangeToNonSelectedMatchingSegmentsInBurstingColumn()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.PermanenceDecrement = 0.08;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            int[] activeColumns = { 1 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            Cell[] burstingCells = { cn.GetCell(4), cn.GetCell(5) };

            DistalDendrite selectedMatchingSegment = cn.CreateDistalSegment(burstingCells[0]);
            cn.CreateSynapse(selectedMatchingSegment, previousActiveCells[0], 0.3);
            cn.CreateSynapse(selectedMatchingSegment, previousActiveCells[1], 0.3);
            cn.CreateSynapse(selectedMatchingSegment, previousActiveCells[2], 0.3);
            cn.CreateSynapse(selectedMatchingSegment, cn.GetCell(81), 0.3);

            DistalDendrite otherMatchingSegment = cn.CreateDistalSegment(burstingCells[1]);
            Synapse as1 = cn.CreateSynapse(otherMatchingSegment, previousActiveCells[0], 0.3);
            Synapse as2 = cn.CreateSynapse(otherMatchingSegment, previousActiveCells[1], 0.3);
            Synapse is1 = cn.CreateSynapse(otherMatchingSegment, cn.GetCell(81), 0.3);

            tm.Compute(previousActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0.3, as1.Permanence, 0.01);
            Assert.AreEqual(0.3, as2.Permanence, 0.01);
            Assert.AreEqual(0.3, is1.Permanence, 0.01);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestNoChangeToMatchingSegmentsInPredictedActiveColumn()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            int[] activeColumns = { 1 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            Cell expectedActiveCell = cn.GetCell(4);
            List<Cell> expectedActiveCells = new List<Cell>(new Cell[] { expectedActiveCell });
            Cell otherBurstingCell = cn.GetCell(5);

            DistalDendrite activeSegment = cn.CreateDistalSegment(expectedActiveCell);
            cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[1], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[2], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[3], 0.5);

            DistalDendrite matchingSegmentOnSameCell = cn.CreateDistalSegment(expectedActiveCell);
            Synapse s1 = cn.CreateSynapse(matchingSegmentOnSameCell, previousActiveCells[0], 0.3);
            Synapse s2 = cn.CreateSynapse(matchingSegmentOnSameCell, previousActiveCells[1], 0.3);

            DistalDendrite matchingSegmentOnOtherCell = cn.CreateDistalSegment(otherBurstingCell);
            Synapse s3 = cn.CreateSynapse(matchingSegmentOnOtherCell, previousActiveCells[0], 0.3);
            Synapse s4 = cn.CreateSynapse(matchingSegmentOnOtherCell, previousActiveCells[1], 0.3);

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            Assert.IsTrue(cc.PredictiveCells.SequenceEqual(expectedActiveCells));
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0.3, s1.Permanence, 0.01);
            Assert.AreEqual(0.3, s2.Permanence, 0.01);
            Assert.AreEqual(0.3, s3.Permanence, 0.01);
            Assert.AreEqual(0.3, s4.Permanence, 0.01);
        }


        [TestMethod]
        [TestCategory("Prod")]
        public void TestNoNewSegmentIfNotEnoughWinnerCells()
        {
            TemporalMemory tm = new TemporalMemory();

            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.MaxNewSynapseCount = 3;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] zeroColumns = { };
            int[] activeColumns = { 0 };

            tm.Compute(zeroColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0, cn.NumSegments(), 0);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestNewSegmentAddSynapsesToSubsetOfWinnerCells()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.MaxNewSynapseCount = 2;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0, 1, 2 };
            int[] activeColumns = { 4 };

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;

            IList<Cell> prevWinnerCells = cc.WinnerCells;
            Assert.AreEqual(3, prevWinnerCells.Count);

            cc = tm.Compute(activeColumns, true) as ComputeCycle;

            List<Cell> winnerCells = new List<Cell>(cc.WinnerCells);
            Assert.AreEqual(1, winnerCells.Count);

            List<DistalDendrite> segments = winnerCells[0].GetSegments(cn);
            //List<DistalDendrite> segments = winnerCells[0].Segments;
            Assert.AreEqual(1, segments.Count);

            List<Synapse> synapses = cn.GetSynapses(segments[0]);
            Assert.AreEqual(2, synapses.Count);

            foreach (Synapse synapse in synapses)
            {
                Assert.AreEqual(0.21, synapse.Permanence, 0.01);
                Assert.IsTrue(prevWinnerCells.Contains(synapse.getPresynapticCell()));
            }
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestNewSegmentAddSynapsesToAllWinnerCells()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.MaxNewSynapseCount = 4;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0, 1, 2 };
            int[] activeColumns = { 4 };

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            List<Cell> prevWinnerCells = new List<Cell>(cc.WinnerCells);
            Assert.AreEqual(3, prevWinnerCells.Count);

            cc = tm.Compute(activeColumns, true) as ComputeCycle;

            List<Cell> winnerCells = new List<Cell>(cc.WinnerCells);
            Assert.AreEqual(1, winnerCells.Count);
            List<DistalDendrite> segments = winnerCells[0].GetSegments(cn);
            //List<DistalDendrite> segments = winnerCells[0].Segments;
            Assert.AreEqual(1, segments.Count);
            List<Synapse> synapses = segments[0].GetAllSynapses(cn);

            List<Cell> presynapticCells = new List<Cell>();
            foreach (Synapse synapse in synapses)
            {
                Assert.AreEqual(0.21, synapse.Permanence, 0.01);
                presynapticCells.Add(synapse.getPresynapticCell());
            }

            presynapticCells.Sort();

            Assert.IsTrue(prevWinnerCells.SequenceEqual(presynapticCells));
        }

        [TestMethod]
        public void TestMatchingSegmentAddSynapsesToSubsetOfWinnerCells()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.CellsPerColumn = 1;
            htmConfig.MinThreshold = 1;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0, 1, 2, 3 };
            IList<Cell> prevWinnerCells = cn.GetCellSet(new int[] { 0, 1, 2, 3 });
            int[] activeColumns = { 4 };

            DistalDendrite matchingSegment = cn.CreateDistalSegment(cn.GetCell(4));
            cn.CreateSynapse(matchingSegment, cn.GetCell(0), 0.5);

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            Assert.IsTrue(cc.WinnerCells.SequenceEqual(prevWinnerCells));
            cc = tm.Compute(activeColumns, true) as ComputeCycle;

            List<Synapse> synapses = cn.GetSynapses(matchingSegment);
            Assert.AreEqual(3, synapses.Count);

            synapses.Sort();
            foreach (Synapse synapse in synapses)
            {
                if (synapse.getPresynapticCell().Index == 0) continue;

                Assert.AreEqual(0.21, synapse.Permanence, 0.01);
                Assert.IsTrue(synapse.getPresynapticCell().Index == 1 ||
                           synapse.getPresynapticCell().Index == 2 ||
                           synapse.getPresynapticCell().Index == 3);
            }
        }


        [TestMethod]
        [TestCategory("Prod")]
        public void TestMatchingSegmentAddSynapsesToAllWinnerCells()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.CellsPerColumn = 1;
            htmConfig.MinThreshold = 1;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0, 1 };
            IList<Cell> prevWinnerCells = cn.GetCellSet(new int[] { 0, 1 });
            int[] activeColumns = { 4 };

            DistalDendrite matchingSegment = cn.CreateDistalSegment(cn.GetCell(4));
            cn.CreateSynapse(matchingSegment, cn.GetCell(0), 0.5);

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            Assert.IsTrue(cc.WinnerCells.SequenceEqual(prevWinnerCells));

            cc = tm.Compute(activeColumns, true) as ComputeCycle;

            List<Synapse> synapses = cn.GetSynapses(matchingSegment);
            Assert.AreEqual(2, synapses.Count);

            synapses.Sort();

            foreach (Synapse synapse in synapses)
            {
                if (synapse.getPresynapticCell().Index == 0) continue;

                Assert.AreEqual(0.21, synapse.Permanence, 0.01);
                Assert.AreEqual(1, synapse.getPresynapticCell().Index);
            }
        }

        /**
         * When a segment becomes active, grow synapses to previous winner cells.
         *
         * The number of grown synapses is calculated from the "matching segment"
         * overlap, not the "active segment" overlap.
         */
        [TestMethod]
        public void TestActiveSegmentGrowSynapsesAccordingToPotentialOverlap()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.CellsPerColumn = 1;
            htmConfig.MinThreshold = 1;
            htmConfig.ActivationThreshold = 2;
            htmConfig.MaxNewSynapseCount = 4;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            // Use 1 cell per column so that we have easy control over the winner cells.
            int[] previousActiveColumns = { 0, 1, 2, 3, 4 };
            List<Cell> prevWinnerCells = new List<Cell>(new Cell[] { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3), cn.GetCell(4) });

            int[] activeColumns = { 5 };

            DistalDendrite activeSegment = cn.CreateDistalSegment(cn.GetCell(5));
            cn.CreateSynapse(activeSegment, cn.GetCell(0), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(1), 0.5);
            cn.CreateSynapse(activeSegment, cn.GetCell(2), 0.2);

            ComputeCycle cc = tm.Compute(previousActiveColumns, true) as ComputeCycle;
            Assert.IsTrue(prevWinnerCells.SequenceEqual(cc.WinnerCells));
            cc = tm.Compute(activeColumns, true) as ComputeCycle;

            List<Cell> presynapticCells = new List<Cell>();
            foreach (var syn in activeSegment.GetAllSynapses(cn))
            {
                presynapticCells.Add(syn.getPresynapticCell());
            }

            //= cn.getSynapses(activeSegment).stream()
            //.map(s->s.getPresynapticCell())
            //.collect(Collectors.toSet());

            Assert.IsTrue(
                presynapticCells.Count == 4 && (
                (presynapticCells.Contains(cn.GetCell(0)) && presynapticCells.Contains(cn.GetCell(1)) && presynapticCells.Contains(cn.GetCell(2)) && presynapticCells.Contains(cn.GetCell(3))) ||
                (presynapticCells.Contains(cn.GetCell(0)) && presynapticCells.Contains(cn.GetCell(1)) && presynapticCells.Contains(cn.GetCell(2)) && presynapticCells.Contains(cn.GetCell(4)))));
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestDestroyWeakSynapseOnWrongPrediction()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.InitialPermanence = 0.2;
            htmConfig.MaxNewSynapseCount = 4;
            htmConfig.PredictedSegmentDecrement = 0.02;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            int[] activeColumns = { 2 };
            Cell expectedActiveCell = cn.GetCell(5);

            DistalDendrite activeSegment = cn.CreateDistalSegment(expectedActiveCell);
            cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[1], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[2], 0.5);
            // Weak Synapse
            cn.CreateSynapse(activeSegment, previousActiveCells[3], 0.015);

            tm.Compute(previousActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(3, cn.GetNumSynapses(activeSegment));
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestDestroyWeakSynapseOnActiveReinforce()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.InitialPermanence = 0.2;
            htmConfig.MaxNewSynapseCount = 4;
            htmConfig.PredictedSegmentDecrement = 0.02;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] previousActiveColumns = { 0 };
            Cell[] previousActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            int[] activeColumns = { 2 };
            Cell expectedActiveCell = cn.GetCell(5);

            DistalDendrite activeSegment = cn.CreateDistalSegment(expectedActiveCell);
            cn.CreateSynapse(activeSegment, previousActiveCells[0], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[1], 0.5);
            cn.CreateSynapse(activeSegment, previousActiveCells[2], 0.5);
            // Weak Synapse
            cn.CreateSynapse(activeSegment, previousActiveCells[3], 0.009);

            tm.Compute(previousActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(3, cn.GetNumSynapses(activeSegment));
        }

        [TestMethod]
        public void TestRecycleWeakestSynapseToMakeRoomForNewSynapse()
        {
            throw new AssertInconclusiveException("Not fixed.");

            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.CellsPerColumn = 30;
            htmConfig.ColumnDimensions = new int[] { 100 };
            //htmConfig.ColumnDimensions = new int[] { 5 };
            htmConfig.MinThreshold = 1;
            htmConfig.PermanenceIncrement = 0.02;
            htmConfig.PermanenceDecrement = 0.02;
            htmConfig.MaxSynapsesPerSegment = 3;

            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            Assert.AreEqual(3, cn.HtmConfig.MaxSynapsesPerSegment);

            int[] prevActiveColumns = { 0, 1, 2 };
            IList<Cell> prevWinnerCells = cn.GetCellSet(new int[] { 0, 1, 2 });
            int[] activeColumns = { 4 };

            DistalDendrite matchingSegment = cn.CreateDistalSegment(cn.GetCell(4));
            cn.CreateSynapse(matchingSegment, cn.GetCell(81), 0.6);
            // Weakest Synapse
            cn.CreateSynapse(matchingSegment, cn.GetCell(0), 0.11);

            ComputeCycle cc = tm.Compute(prevActiveColumns, true) as ComputeCycle;
            Assert.IsTrue(prevWinnerCells.SequenceEqual(cc.WinnerCells));
            tm.Compute(activeColumns, true);

            List<Synapse> synapses = cn.GetSynapses(matchingSegment);
            Assert.AreEqual(3, synapses.Count);
            //Set<Cell> presynapticCells = synapses.stream().map(s->s.getPresynapticCell()).collect(Collectors.toSet());
            List<Cell> presynapticCells = new List<Cell>();
            foreach (var syn in cn.GetSynapses(matchingSegment))
            {
                presynapticCells.Add(syn.getPresynapticCell());
            }

            Assert.IsFalse(presynapticCells.Count(c => c.Index == 0) > 0);

            //Assert.IsFalse(presynapticCells.stream().mapToInt(cell->cell.getIndex()).anyMatch(i->i == 0));
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestRecycleLeastRecentlyActiveSegmentToMakeRoomForNewSegment()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.CellsPerColumn = 1;
            htmConfig.InitialPermanence = 0.5;
            htmConfig.PermanenceIncrement = 0.02;
            htmConfig.PermanenceDecrement = 0.02;
            htmConfig.MaxSegmentsPerCell = 2;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] prevActiveColumns1 = { 0, 1, 2 };
            int[] prevActiveColumns2 = { 3, 4, 5 };
            int[] prevActiveColumns3 = { 6, 7, 8 };
            int[] activeColumns = { 9 };
            Cell cell9 = cn.GetCell(9);

            tm.Compute(prevActiveColumns1, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(1, cn.GetSegments(cell9).Count);
            DistalDendrite oldestSegment = cn.GetSegments(cell9)[0];
            tm.Reset(cn);
            tm.Compute(prevActiveColumns2, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(2, cn.GetSegments(cell9).Count);

            //Set<Cell> oldPresynaptic = cn.getSynapses(oldestSegment)
            //    .stream()
            //    .map(s->s.getPresynapticCell())
            //    .collect(Collectors.toSet());

            var oldPresynaptic = cn.GetSynapses(oldestSegment).Select(s => s.getPresynapticCell()).ToList();

            tm.Reset(cn);
            tm.Compute(prevActiveColumns3, true);
            tm.Compute(activeColumns, true);
            Assert.AreEqual(2, cn.GetSegments(cell9).Count);

            // Verify none of the segments are connected to the cells the old
            // segment was connected to.

            foreach (DistalDendrite segment in cn.GetSegments(cell9))
            {
                //Set<Cell> newPresynaptic = cn.getSynapses(segment)
                //    .stream()
                //    .map(s->s.getPresynapticCell())
                //    .collect(Collectors.toSet());
                var newPresynaptic = cn.GetSynapses(segment).Select(s => s.getPresynapticCell()).ToList();


                Assert.IsTrue(areDisjoined<Cell>(oldPresynaptic, newPresynaptic));
            }
        }


        [TestMethod]
        [TestCategory("Prod")]
        public void TestDestroySegmentsWithTooFewSynapsesToBeMatching()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.InitialPermanence = 0.2;
            htmConfig.MaxNewSynapseCount = 4;
            htmConfig.PredictedSegmentDecrement = 0.02;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] prevActiveColumns = { 0 };
            Cell[] prevActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            int[] activeColumns = { 2 };
            Cell expectedActiveCell = cn.GetCell(5);

            DistalDendrite matchingSegment = cn.CreateDistalSegment(cn.GetCell(5));
            cn.CreateSynapse(matchingSegment, prevActiveCells[0], .015);
            cn.CreateSynapse(matchingSegment, prevActiveCells[1], .015);
            cn.CreateSynapse(matchingSegment, prevActiveCells[2], .015);
            cn.CreateSynapse(matchingSegment, prevActiveCells[3], .015);

            tm.Compute(prevActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0, cn.NumSegments(expectedActiveCell));
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestPunishMatchingSegmentsInInactiveColumns()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.MaxNewSynapseCount = 4;
            htmConfig.InitialPermanence = 0.2;
            htmConfig.PredictedSegmentDecrement = 0.02;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] prevActiveColumns = { 0 };
            Cell[] prevActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            int[] activeColumns = { 1 };
            Cell previousInactiveCell = cn.GetCell(81);

            DistalDendrite activeSegment = cn.CreateDistalSegment(cn.GetCell(42));
            Synapse as1 = cn.CreateSynapse(activeSegment, prevActiveCells[0], .5);
            Synapse as2 = cn.CreateSynapse(activeSegment, prevActiveCells[1], .5);
            Synapse as3 = cn.CreateSynapse(activeSegment, prevActiveCells[2], .5);
            Synapse is1 = cn.CreateSynapse(activeSegment, previousInactiveCell, .5);

            DistalDendrite matchingSegment = cn.CreateDistalSegment(cn.GetCell(43));
            Synapse as4 = cn.CreateSynapse(matchingSegment, prevActiveCells[0], .5);
            Synapse as5 = cn.CreateSynapse(matchingSegment, prevActiveCells[1], .5);
            Synapse is2 = cn.CreateSynapse(matchingSegment, previousInactiveCell, .5);

            tm.Compute(prevActiveColumns, true);
            tm.Compute(activeColumns, true);

            Assert.AreEqual(0.48, as1.Permanence, 0.01);
            Assert.AreEqual(0.48, as2.Permanence, 0.01);
            Assert.AreEqual(0.48, as3.Permanence, 0.01);
            Assert.AreEqual(0.48, as4.Permanence, 0.01);
            Assert.AreEqual(0.48, as5.Permanence, 0.01);
            Assert.AreEqual(0.50, is1.Permanence, 0.01);
            Assert.AreEqual(0.50, is2.Permanence, 0.01);
        }

        [TestMethod]
        public void TestAddSegmentToCellWithFewestSegments()
        {
            bool grewOnCell1 = false;
            bool grewOnCell2 = false;

            for (int seed = 0; seed < 100; seed++)
            {
                TemporalMemory tm = new TemporalMemory();
                HtmConfig htmConfig = GetDefaultTMParameters();
                htmConfig.MaxNewSynapseCount = 4;
                htmConfig.PredictedSegmentDecrement = 0.02;
                htmConfig.RandomGenSeed = seed;
                Connections cn = new Connections(htmConfig);
                tm.Init(cn);

                int[] prevActiveColumns = { 1, 2, 3, 4 };
                Cell[] prevActiveCells = { cn.GetCell(4), cn.GetCell(5), cn.GetCell(6), cn.GetCell(7) };
                int[] activeColumns = { 0 };
                Cell[] nonMatchingCells = { cn.GetCell(0), cn.GetCell(3) };
                IList<Cell> activeCells = cn.GetCellSet(new int[] { 0, 1, 2, 3 });

                DistalDendrite segment1 = cn.CreateDistalSegment(nonMatchingCells[0]);
                cn.CreateSynapse(segment1, prevActiveCells[0], 0.5);
                DistalDendrite segment2 = cn.CreateDistalSegment(nonMatchingCells[1]);
                cn.CreateSynapse(segment2, prevActiveCells[1], 0.5);

                tm.Compute(prevActiveColumns, true);
                ComputeCycle cc = tm.Compute(activeColumns, true) as ComputeCycle;

                Assert.IsTrue(cc.ActiveCells.SequenceEqual(activeCells));

                Assert.AreEqual(3, cn.NumSegments());
                Assert.AreEqual(1, cn.NumSegments(cn.GetCell(0)));
                Assert.AreEqual(1, cn.NumSegments(cn.GetCell(3)));
                Assert.AreEqual(1, cn.GetNumSynapses(segment1));
                Assert.AreEqual(1, cn.GetNumSynapses(segment2));

                List<DistalDendrite> segments = new List<DistalDendrite>(cn.GetSegments(cn.GetCell(1)));
                if (segments.Count == 0)
                {
                    List<DistalDendrite> segments2 = cn.GetSegments(cn.GetCell(2));
                    Assert.IsFalse(segments2.Count == 0);
                    grewOnCell2 = true;
                    segments.AddRange(segments2);
                }
                else
                {
                    grewOnCell1 = true;
                }

                Assert.AreEqual(1, segments.Count);
                List<Synapse> synapses = segments[0].GetAllSynapses(cn);
                Assert.AreEqual(4, synapses.Count);

                ISet<Column> columnCheckList = cn.GetColumnSet(prevActiveColumns);

                foreach (Synapse synapse in synapses)
                {
                    Assert.AreEqual(0.2, synapse.Permanence, 0.01);

                    var parentColIndx = synapse.getPresynapticCell().ParentColumnIndex;
                    Column column = cn.HtmConfig.Memory.GetColumn(parentColIndx);
                    Assert.IsTrue(columnCheckList.Contains(column));
                    columnCheckList.Remove(column);
                }

                Assert.AreEqual(0, columnCheckList.Count);
            }

            Assert.IsTrue(grewOnCell1);
            Assert.IsTrue(grewOnCell2);
        }

        [TestMethod]
        [TestCategory("Tests with Serialization Issue")]
        public void TestConnectionsNeverChangeWhenLearningDisabled()
        {
            throw new AssertInconclusiveException("Not fixed.");

            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.MaxNewSynapseCount = 4;
            htmConfig.PredictedSegmentDecrement = 0.02;
            htmConfig.InitialPermanence = 0.2;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            int[] prevActiveColumns = { 0 };
            Cell[] prevActiveCells = { cn.GetCell(0), cn.GetCell(1), cn.GetCell(2), cn.GetCell(3) };
            int[] activeColumns = { 1, 2 };
            Cell prevInactiveCell = cn.GetCell(81);
            Cell expectedActiveCell = cn.GetCell(4);

            DistalDendrite correctActiveSegment = cn.CreateDistalSegment(expectedActiveCell);
            cn.CreateSynapse(correctActiveSegment, prevActiveCells[0], 0.5);
            cn.CreateSynapse(correctActiveSegment, prevActiveCells[1], 0.5);
            cn.CreateSynapse(correctActiveSegment, prevActiveCells[2], 0.5);

            DistalDendrite wrongMatchingSegment = cn.CreateDistalSegment(cn.GetCell(43));
            cn.CreateSynapse(wrongMatchingSegment, prevActiveCells[0], 0.5);
            cn.CreateSynapse(wrongMatchingSegment, prevActiveCells[1], 0.5);
            cn.CreateSynapse(wrongMatchingSegment, prevInactiveCell, 0.5);

            var r = deepCopyPlain<Synapse>(cn.getReceptorSynapseMapping().Values.First().First());
            var synMapBefore = deepCopyPlain<Dictionary<Cell, LinkedHashSet<Synapse>>>(cn.getReceptorSynapseMapping());
            var segMapBefore = deepCopyPlain<Dictionary<Cell, List<DistalDendrite>>>(cn.GetSegmentMapping());

            tm.Compute(prevActiveColumns, false);
            tm.Compute(activeColumns, false);

            Assert.IsTrue(synMapBefore != cn.getReceptorSynapseMapping());
            Assert.IsTrue(synMapBefore.Keys.SequenceEqual(cn.getReceptorSynapseMapping().Keys));

            Assert.IsTrue(segMapBefore != cn.GetSegmentMapping());
            Assert.IsTrue(segMapBefore.Keys.SequenceEqual(cn.GetSegmentMapping().Keys));
        }

        public void TestLeastUsedCell()
        {
            TemporalMemory tm = new TemporalMemory();
            HtmConfig htmConfig = GetDefaultTMParameters();
            htmConfig.ColumnDimensions = new int[] { 2 };
            htmConfig.CellsPerColumn = 2;
            Connections cn = new Connections(htmConfig);
            tm.Init(cn);

            DistalDendrite dd = cn.CreateDistalSegment(cn.GetCell(0));
            cn.CreateSynapse(dd, cn.GetCell(3), 0.3);

            for (int i = 0; i < 100; i++)
            {
                Assert.AreEqual(1, tm.GetLeastUsedCell(cn, cn.GetColumn(0).Cells, cn.HtmConfig.Random).Index);
            }
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestAdaptSegment()
        {
            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            Parameters p = Parameters.getAllDefaultParameters();
            p.apply(cn);
            tm.Init(cn);

            DistalDendrite dd = cn.CreateDistalSegment(cn.GetCell(0));
            Synapse s1 = cn.CreateSynapse(dd, cn.GetCell(23), 0.6);
            Synapse s2 = cn.CreateSynapse(dd, cn.GetCell(37), 0.4);
            Synapse s3 = cn.CreateSynapse(dd, cn.GetCell(477), 0.9);

            tm.AdaptSegment(cn, dd, cn.GetCellSet(new int[] { 23, 37 }), cn.HtmConfig.PermanenceIncrement, cn.HtmConfig.PermanenceDecrement);

            Assert.AreEqual(0.7, s1.Permanence, 0.01);
            Assert.AreEqual(0.5, s2.Permanence, 0.01);
            Assert.AreEqual(0.8, s3.Permanence, 0.01);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestAdaptSegmentToMax()
        {
            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            Parameters p = Parameters.getAllDefaultParameters();
            p.apply(cn);
            tm.Init(cn);

            DistalDendrite dd = cn.CreateDistalSegment(cn.GetCell(0));
            Synapse s1 = cn.CreateSynapse(dd, cn.GetCell(23), 0.9);

            tm.AdaptSegment(cn, dd, cn.GetCellSet(new int[] { 23 }), cn.HtmConfig.PermanenceIncrement, cn.HtmConfig.PermanenceDecrement);
            Assert.AreEqual(1.0, s1.Permanence, 0.1);

            // Now permanence should be at max
            tm.AdaptSegment(cn, dd, cn.GetCellSet(new int[] { 23 }), cn.HtmConfig.PermanenceIncrement, cn.HtmConfig.PermanenceDecrement);
            Assert.AreEqual(1.0, s1.Permanence, 0.1);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void TestAdaptSegmentToMin()
        {
            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            Parameters p = Parameters.getAllDefaultParameters();
            p.apply(cn);
            tm.Init(cn);

            DistalDendrite dd = cn.CreateDistalSegment(cn.GetCell(0));
            Synapse s1 = cn.CreateSynapse(dd, cn.GetCell(23), 0.1);
            cn.CreateSynapse(dd, cn.GetCell(1), 0.3);

            tm.AdaptSegment(cn, dd, cn.GetCellSet(new int[] { }), cn.HtmConfig.PermanenceIncrement, cn.HtmConfig.PermanenceDecrement);
            Assert.IsFalse(cn.GetSynapses(dd).Contains(s1));
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void testNumberOfColumns()
        {
            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            Parameters p = Parameters.getAllDefaultParameters();
            p.Set(KEY.COLUMN_DIMENSIONS, new int[] { 64, 64 });
            p.Set(KEY.CELLS_PER_COLUMN, 32);
            p.apply(cn);
            tm.Init(cn);

            Assert.AreEqual(64 * 64, cn.HtmConfig.NumColumns);
        }

        [TestMethod]
        [TestCategory("Prod")]
        public void testNumberOfCells()
        {
            TemporalMemory tm = new TemporalMemory();
            Connections cn = new Connections();
            Parameters p = Parameters.getAllDefaultParameters();
            p.Set(KEY.COLUMN_DIMENSIONS, new int[] { 64, 64 });
            p.Set(KEY.CELLS_PER_COLUMN, 32);
            p.apply(cn);
            tm.Init(cn);

            Assert.AreEqual(64 * 64 * 32, cn.Cells.Length);
        }

        public void TemporalMemoryInit()
        {
            HtmConfig htmConfig = new HtmConfig();
            Connections connections = new Connections(htmConfig);

            TemporalMemory temporalMemory = new TemporalMemory();

            temporalMemory.Init(connections);
        }
    }
}
