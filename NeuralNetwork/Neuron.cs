using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class Neuron
    {
        public enum Type { input, hidden, output };
        Type t;
        double a;//value of a neuron
        double z;//input of a neuron
        double delta;
        double gPrime;
        public Neuron()
        { }
        public Neuron(Neuron N)
        {
            this.t=N.t;
            this.a = N.a;
            this.z = N.Z;
            this.delta = N.delta;
        }
        public double A 
        {
            get { return a; }
            set { a = value; }
        }
        public double GPrime
        {
            get { return gPrime; }
            set { gPrime = value; }
        }
        public double Z
        {
            get { return z; }
            set { z = value; }
        }
        public Type T
        {
            get { return t; }
            set { t = value; }
        }
        public double Delta
        {
            get { return delta; }
            set { delta = value; }
        }
        public void g(double _inputValue)
        {
            z = _inputValue;
            if (t == Type.hidden)
                a = sigmoid(z);
            if (t == Type.output)
                a = a = sigmoid(z);
            if (t == Type.input)
                a = z;
        }
        public void gPrimeFunction()
        {
            if (t == Type.hidden)
                gPrime = sigmoid(z) * (1 - sigmoid(z));
            if (t == Type.output)
                gPrime = sigmoid(z) * (1 - sigmoid(z));
            if (t == Type.input)
                gPrime = sigmoid(z) * (1 - sigmoid(z));
        }
        private double sigmoid(double vale)
        {
            return 1.0 / (1.0 + Math.Exp(-vale));
        }
        
    }
}
