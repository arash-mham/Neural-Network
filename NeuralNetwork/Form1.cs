using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            int[] a={2,2,1};
            Network N = new Network(3, a);
            List<double>input=new List<double> ();
            input.Add(0);
            input.Add(1);
            N.receiveInput(input);
            N.feedForward_H_Theta();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }
    }
}
