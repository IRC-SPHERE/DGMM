//
// MoG.cs
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
    public static class Models
    {
        public static void MixtureOfGaussians(Vector[] x, int numberOfComponents)
        {
            var numExamples = x.Length;
            var numDims = x[0].Count;
            var numComponents = Variable.Observed(numberOfComponents).Named("numberOfComponents");

            var component = new Range(numComponents).Named("component");
            var example = new Range(numExamples).Named("numExamples");

            var weights = Variable.DirichletUniform(component);

            // Mixture component means
            var means = Variable.Array<Vector>(component).Named("means");         
            var precs = Variable.Array<PositiveDefiniteMatrix>(component).Named("precs");

            double scale = 1;

            means[component] = Variable.VectorGaussianFromMeanAndPrecision(
                Vector.Constant(numDims, 0.0),
                PositiveDefiniteMatrix.IdentityScaledBy(numDims, scale)).ForEach(component);

            precs[component] = Variable.WishartFromShapeAndScale(
                1.0/scale, PositiveDefiniteMatrix.IdentityScaledBy(numDims, scale)).ForEach(component);

            // Create a variable array which will hold the data
            var data = Variable.Array<Vector>(example).Named("x");

            // Create latent indicator variable for each data point
            var z = Variable.Array<int>(example).Named("z");

            // The mixture of Gaussians model
            using (Variable.ForEach(example)) 
            {
                z[example] = Variable.Discrete(weights);
                using (Variable.Switch(z[example])) 
                {
                    data[example] = Variable.VectorGaussianFromMeanAndPrecision(
                        means[z[example]], 
                        precs[z[example]]);
                }
            }

            // Attach some generated data
            data.ObservedValue = x;

            // Initialise messages randomly so as to break symmetry
            var zinit = Enumerable.Range(0, numExamples).Select(
                ia => Discrete.PointMass(Rand.Int(numberOfComponents), numberOfComponents)
            ).ToArray();

            z.InitialiseTo(Distribution<int>.Array(zinit)); 

            // The inference
            var ie = new InferenceEngine(new ExpectationPropagation { DefaultNumberOfIterations = 100 });
            Console.WriteLine("Dist over pi=" + ie.Infer(weights));
            Console.WriteLine("Dist over means=\n" + ie.Infer(means));
            Console.WriteLine("Dist over precs=\n" + ie.Infer(precs));
        }

        public static void SwitchOverMixtures(Vector[] x, int[] numberOfComponents)
        {
            var numExamples = x.Length;
            var numDims = x[0].Count;

            // ToDo; better name for loop over numbers of components
            var kk = new Range(numberOfComponents.Length).Named("kk");
            //var ks = Variable.Discrete(Vector.Constant(numberOfComponents.Length, 1.0 / numberOfComponents.Length)).Named("ks");
            //ks.SetValueRange(kk);

            var numComponents = Variable.Array<int>(kk).Named("numberOfComponents");

            var example = new Range(numExamples).Named("numExamples");

            var component = new Range(numComponents[kk]).Named("component");

            var weights = Variable.DirichletUniform(component);
            var outerWeights = Variable.DirichletUniform(kk).Named("outerWeights");

            var ks = Variable.Discrete(outerWeights).Named("ks");

            // Mixture component means
            var means = Variable.Array(Variable.Array<Vector>(component), kk).Named("means");         
            var precs = Variable.Array(Variable.Array<PositiveDefiniteMatrix>(component), kk).Named("precs");

            // Create latent indicator variable for each data point
            var z = Variable.Array(Variable.Array<int>(example), kk).Named("z");

            using (Variable.ForEach(kk))
            {
                double scale = 0.01;

                means[kk][component] = Variable.VectorGaussianFromMeanAndPrecision(
                    Vector.Constant(numDims, 0.0),
                    PositiveDefiniteMatrix.IdentityScaledBy(numDims, scale)).ForEach(component);

                precs[kk][component] = Variable.WishartFromShapeAndScale(
                    1.0/scale, PositiveDefiniteMatrix.IdentityScaledBy(numDims, scale)).ForEach(component);
                
                using (Variable.ForEach(example))
                {
                    z[kk][example] = Variable.Discrete(weights);
                }
            }
            

            using (var block = Variable.Switch(ks))
            {

                // Create a variable array which will hold the data
                var data = Variable.Array<Vector>(example).Named("x");

                // The mixture of Gaussians model
                using (Variable.ForEach(example)) 
                {
                    using (Variable.Switch(z[ks][example])) 
                    {
                        data[example] = Variable.VectorGaussianFromMeanAndPrecision(
                            means[ks][z[ks][example]],
                            precs[ks][z[ks][example]]);
                    }
                }

                // Attach some generated data
                data.ObservedValue = x;
            }

            // Initialise messages randomly so as to break symmetry
            var zinit = new Discrete[numberOfComponents.Length][];
            for (int i = 0; i < numberOfComponents.Length; i++)
            {
                zinit[i] = new Discrete[numExamples];
                for (int j = 0; j < numExamples; j++)
                {
                    zinit[i][j] = Discrete.PointMass(Rand.Int(numberOfComponents[i]), numberOfComponents[i]);
                }
            }

            z.InitialiseTo(Distribution<int>.Array(zinit));

            // Also initialise ks randomly
            var ksInit = Discrete.PointMass(Rand.Int(numberOfComponents.Length), numberOfComponents.Length);
            ks.InitialiseTo(ksInit);

            numComponents.ObservedValue = numberOfComponents;

            // The inference
            //var ie = new InferenceEngine(new ExpectationPropagation { DefaultNumberOfIterations = 100 });
            var ie = new InferenceEngine(new GibbsSampling { DefaultNumberOfIterations = 1000, BurnIn = 200, Thin = 200 });
            //Console.WriteLine("Dist over ks=" + ie.Infer(ks));
            Console.WriteLine("Dist over phi=" + ie.Infer(outerWeights));
            //Console.WriteLine("Dist over pi=" + ie.Infer(weights));
            //Console.WriteLine("Dist over means=\n" + ie.Infer(means));
            //Console.WriteLine("Dist over precs=\n" + ie.Infer(precs));
        }
    }
}

