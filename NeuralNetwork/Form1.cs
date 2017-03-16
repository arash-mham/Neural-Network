using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Tao.OpenGl;


namespace NeuralNetwork
{
    public partial class Form1 : Form
    {
       
        Network N = new Network();
        public Form1()
        {
            
            InitializeComponent();
            simpleOpenGlControl1.InitializeContexts();
            trainNetwork();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void simpleOpenGlControl1_Load(object sender, EventArgs e)
        {

        }

        private void simpleOpenGlControl1_Paint(object sender, PaintEventArgs e)
        {
            plot();
        }
        private void trainNetwork()
        {
            int[] a = { 2, 5, 2 };
            N = new Network(3, a);

            Random rand = new Random();

            for (int i = 0; i < 200000; i++)
            {

                int rnd = rand.Next(0, 100);
                double x = 1 - (double)rnd / 50.0;
                rnd = rand.Next(0, 1000);
                double y = 1 - (double)rnd / 500.0;
                List<double> input = new List<double>();
                input.Add(x);
                input.Add(y);

                N.receiveInput(input);
                N.feedForward_H_Theta();

                input.Clear();


                if (Math.Pow(x,1) > y)//you can define any function here and see the plot
                {
                    input.Add(1);
                    input.Add(0);
                }
                else
                {
                    input.Add(0);
                    input.Add(1);
                }

                N.assignOutput(input);
                N.backPropagation();
            }
 
        }
        private void plot()
        {
            //sample a grid and ask its type from the network
             for (int i = 0; i < 100; i++)
            {
                for (int j = 0; j < 100; j++)
                {
                    List<double> test_input = new List<double>();
                    test_input.Add(1-(double)i/50.0);
                    test_input.Add(1-(double)j /50.0);
                    N.receiveInput(test_input);
                    N.feedForward_H_Theta();
                    Gl.glPointSize(4);
                    if (N.getNeuron(2, 0).A > N.getNeuron(2, 1).A)
                        Gl.glColor3f(1,0,0);
                    else
                        Gl.glColor3f(0, 0, 1);
                    Gl.glBegin(Gl.GL_POINTS);
                        Gl.glVertex2d(test_input[0],test_input[1]);
                    Gl.glEnd();


                }
            }
            //draw the origin
            Gl.glColor3f(0, 1, 0);
            Gl.glBegin(Gl.GL_POINTS);
            Gl.glVertex2d(0,0);
            Gl.glEnd();

        }


    }
}
