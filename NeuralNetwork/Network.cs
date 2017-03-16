using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading;

namespace NeuralNetwork
{
    class Network
    {
        List<List<Neuron>> neuronList = new List<List<Neuron>>();
        List<List<List<double>>> thetaList = new List<List<List<double>>>();
        List<double> outPutList = new List<double>();
        int layerNumber;
        public Network()
        { }
        public Network(int _layerNumber, int[] nodeNumebr)
        {
            layerNumber = _layerNumber;
            neuronList.Clear();
            thetaList.Clear();
            if (nodeNumebr.Length != layerNumber)
            {
                MessageBox.Show("The number of layers is not matched!");
                return;
            }
            initializeNeuronList(layerNumber, nodeNumebr);
            initializeThetaList();
        }
        private void initializeNeuronList(int _layerNumber, int[] nodeNumebr)
        {
            for (int i = 0; i < _layerNumber; i++)
            {
                List<Neuron> neuronTempList = new List<Neuron>();
                Random rnd = new Random();
                for (int j = 0; j < nodeNumebr[i]; j++)
                {
                    Thread.Sleep(50);
                    Neuron N = new Neuron();
                    if (i == 0)
                        N.T = Neuron.Type.input;
                    else if (i == layerNumber - 1)
                        N.T = Neuron.Type.output;
                    else
                        N.T = Neuron.Type.hidden;
                    int rand = rnd.Next(1, 100);
                    N.B = (double)(rand / 100.0);
                    neuronTempList.Add(N);
                }
                neuronList.Add(neuronTempList);
            }
        }
        private void initializeThetaList()
        {
            for (int i = 0; i < neuronList.Count - 1; i++)
            {
                List<List<double>> twoDTheta = new List<List<double>>();
                for (int j = 0; j < neuronList[i].Count; j++)
                {
                    List<double> oneDTheta = new List<double>();
                    Random rnd = new Random();
                    for (int k = 0; k < neuronList[i + 1].Count; k++)
                    {

                        Thread.Sleep(50);
                        int rand = rnd.Next(1, 100);
                        oneDTheta.Add((double)(rand/100.0));
                    }
                    twoDTheta.Add(oneDTheta);
                }
                thetaList.Add(twoDTheta);
            }
        }
        public double getTheta(int layer, int indexi, int indexj)
        {
            try
            {
                return thetaList[layer][indexi][indexj];
            }
            catch (System.IndexOutOfRangeException e)
            {
                throw new System.ArgumentOutOfRangeException("index of theta is out of range.", e);
            }
        }
        public void setTheta(int layer, int indexi, int indexj, double value)
        {
            try
            {
                thetaList[layer][indexi][indexj] = value;
            }
            catch (System.IndexOutOfRangeException e)
            {
                throw new System.ArgumentOutOfRangeException("index of theta is out of range.", e);
            }
        }
        public Neuron getNeuron(int layer, int indexi)
        {
            try
            {
                return neuronList[layer][indexi];
            }
            catch (System.IndexOutOfRangeException e)
            {
                throw new System.ArgumentOutOfRangeException("index of neuronList is out of range.", e);
            }
        }
        public void setNeurn(int layer, int indexi, Neuron N)
        {
            try
            {
                neuronList[layer][indexi] = new Neuron(N);
            }
            catch (System.IndexOutOfRangeException e)
            {
                throw new System.ArgumentOutOfRangeException("index of neuronList is out of range.", e);
            }
        }
        public void receiveInput(List<double> inputList)
        {
            if (neuronList.Count == 0 || layerNumber==0)
            {
                MessageBox.Show("Network has not been initialized");
                return;
            }
            if (inputList.Count != neuronList[0].Count)
            {
                MessageBox.Show("inputList and the first layer of the network are not compatible");
                return;
            }
            for (int i = 0; i < inputList.Count; i++)
            {
                neuronList[0][i].Z = inputList[i];
                neuronList[0][i].T = Neuron.Type.input;
                neuronList[0][i].g(inputList[i]);
                neuronList[0][i].gPrimeFunction();
            }

        }
        public void assignOutput(List<double> valueList)
        {
            outPutList.Clear();
            if (neuronList.Count == 0 || layerNumber == 0)
            {
                MessageBox.Show("Network has not been initialized");
                return;
            }
            if (valueList.Count != neuronList[neuronList.Count - 1].Count)
            {
                MessageBox.Show("output list and the last layer of the network are not compatible");
                return;
            }
            outPutList = new List<double>(valueList);

        }
        public void feedForward_H_Theta()
        {
                for (int j = 0; j < neuronList[0].Count; j++)
                {
                    neuronList[0][j].g(neuronList[0][j].Z);
                }
            for (int i = 0; i < layerNumber - 1; i++)
            {
                for (int j = 0; j < neuronList[i + 1].Count; j++)
                {
                    double value = 0;
                    for (int k = 0; k < neuronList[i].Count; k++)
                    {
                        value += thetaList[i][k][j] * neuronList[i][k].A;
                    }
                    neuronList[i + 1][j].g(value + neuronList[i + 1][j].B);
                }
            }

        }
        public void backPropagation()
        {
            outPutLayerDeltaCalculation();
            hiddenLayerDeltaCalculation();
            accumulateGradient();
            
        }
        private void outPutLayerDeltaCalculation()
        {
            for (int i = 0; i < neuronList[layerNumber - 1].Count; i++)
                neuronList[layerNumber - 1][i].Delta = (outPutList[i] - neuronList[layerNumber - 1][i].A) * neuronList[layerNumber - 1][i].A * (1 - neuronList[layerNumber - 1][i].A);//the last row delta
                //neuronList[layerNumber - 1][i].Delta = neuronList[layerNumber - 1][i].A - outPutList[i];//the last row delta
        }
        private void hiddenLayerDeltaCalculation()
        {
            for (int i = layerNumber - 2; i > -1; i--)
            {
                for (int j = 0; j < neuronList[i].Count; j++)
                {
                    double temp = 0;
                    for (int k = 0; k < neuronList[i + 1].Count; k++)
                    {
                       // temp += thetaList[i][j][k]*neuronList[i+1][k].Delta;
                        temp += thetaList[i][j][k] * neuronList[i + 1][k].Delta;

                    }
                   // neuronList[i][j].Delta = temp*neuronList[i][j].GPrime;
                    neuronList[i][j].Delta = temp * (1 - neuronList[i][j].A) * (neuronList[i][j].A);
                }
            }
        }
        private void accumulateGradient()//multiply delta in a
        {
            for (int i = layerNumber - 2; i > -1; i--)
            {
                for (int j = 0; j < neuronList[i].Count; j++)
                {
                    for (int k = 0; k < neuronList[i + 1].Count; k++)
                    {
                        thetaList[i][j][k]+=0.1*( neuronList[i][j].A * neuronList[i+1][k].Delta);
                        neuronList[i][j].B += 0.1*neuronList[i+1][k].Delta;
                    }
                   
                }
            }

            for (int i = layerNumber - 1; i > -1; i--)
            {
                for (int j = 0; j < neuronList[i].Count; j++)
                {
                        neuronList[i][j].B += 0.1 * neuronList[i][j].Delta;
                }
            }
        }
    }
}

