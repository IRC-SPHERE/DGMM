//
// Program.cs
//
// Author:
//       Tom Diethe <tom.diethe@bristol.ac.uk>
//
// Copyright (c) 2016 University of Bristol
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
using System;
using System.Linq;
using MicrosoftResearch.Infer;
using MicrosoftResearch.Infer.Distributions;
using MicrosoftResearch.Infer.Maths;
using MicrosoftResearch.Infer.Models;

namespace DGMM
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            var numExamples = 300;
            var numDims = 2;
            var data = GenerateData(numExamples);
            //Models.MixtureOfGaussians(data, 4);
            Models.SwitchOverMixtures(data, new[] { 1, 2, 3 });
            
        }

        /// <summary>
        /// Generates a data set from a particular true model.
        /// </summary>
        public static Vector[] GenerateData(int nData)
        {
            var trueM1 = Vector.FromArray(2.0, 3.0);
            var trueM2 = Vector.FromArray(7.0, 5.0);
            var trueP1 = new PositiveDefiniteMatrix(
                new double[,] { { 3.0, 0.2 }, { 0.2, 2.0 } });
            var trueP2 = new PositiveDefiniteMatrix(
                new double[,] { { 2.0, 0.4 }, { 0.4, 4.0 } });
            var trueVG1 = VectorGaussian.FromMeanAndPrecision(trueM1, trueP1);
            var trueVG2 = VectorGaussian.FromMeanAndPrecision(trueM2, trueP2);

            double truePi = 0.6;
            var trueB = new Bernoulli(truePi);

            // Restart the infer.NET random number generator
            Rand.Restart(12347);
            var data = new Vector[nData];
            for (int j = 0; j < nData; j++) 
            {
                data[j] = trueB.Sample() ? trueVG1.Sample() : trueVG2.Sample();
            }

            return data;
        }
    }
}
