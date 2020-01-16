using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;

namespace ML
{
    public partial class Form1 : Form
    {
        DateTime dt = DateTime.Now;

        /*
         * Take the input from text file 
         * Also input Number of itterations to stop
         * int samples_ok=0;
         * if sum (d - o )^2 <= mse, then it's ok to continue to the next sample (samples_ok++;)
         * else update all the weights, then it's ok to continuee to the next sample(samples_ok=1;)
         * Will try on Number of Hidden Layers = 2, with number of nodes are 2 and 3 respectively.
         * Will probably give the mse value of 0.6
        */
        public class node
        {
            public float error, value;
            public List<float> inputs = new List<float>();
            public List<float> weights = new List<float>();
            public int Layernum, Nodenum;
        }

        public class layer
        {
            public List<node> nodes = new List<node>();
            public int Layernum;
        }

        int numberOfEnterClicks;
        int numberOfHiddenLayers;
        int otherInputs, howManyGood, desired1, desired2;
        string desired = "";
        float howManyAccurate;
        int ii = 0, itterations = 0, bestHowManyGood;
        float OutputError, HiddenError;

        //Pass the filepath and filename to the StreamWriter
        StreamWriter sw = new StreamWriter("Weights.txt");

        float mse, learnRate, momentum, Errorsqd;
        double e1 = 2.718f;

        Random rand1 = new Random();
        Random rand2 = new Random();
        float RandomNumber;

        List<string> LineSample = new List<string>();

        List<int> numberOfNodes = new List<int>();

        List<layer> tree = new List<layer>();
        List<layer> treeBest = new List<layer>();

        //makee method to calc weight,error for all output and hidden

        public node updateWeights(node x)
        {
            float wnew;
            try
            {
                for (int i = 0; i < x.inputs.Count; i++)
                {
                    wnew = (learnRate * x.error * x.inputs[i]) + (momentum * x.weights[i]);
                    x.weights[i] = wnew;
                    wnew = 0;
                }
            }
            catch (Exception ee)
            {

            }
            return x;
        }
        public float outputError(node x)
        {
            float error1 = 0;
            if (x.Nodenum == 0) { error1 = x.value * (1 - x.value) * (desired1 - x.value); }
            else
            {
                error1 = x.value * (1 - x.value) * (desired2 - x.value);
            }
            return error1;
        }
        public float hidError(node x)
        {
            float error1 = x.value * (1 - x.value);
            float pt2 = 0;
            try
            {
                for (int i = 0; i < tree[x.Layernum + 1].nodes.Count; i++)
                {
                    pt2 += tree[x.Layernum + 1].nodes[i].error * tree[x.Layernum + 1].nodes[i].weights[x.Nodenum];
                }
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
            error1 = error1 * pt2;
            return error1;
        }
        public float calcErrorSqd(node x)
        {
            double doDiff = 0;
            if (x.Nodenum == 0 && x.Layernum == tree.Count - 1)
            {
                doDiff = desired1 - x.value;
            }
            else if (x.Nodenum == 1 && x.Layernum == tree.Count - 1)
            {
                doDiff = desired2 - x.value;
            }
            double errorsqd = Math.Pow(doDiff, 2);

            return (float)errorsqd;
        }
        public float calcSnet(node x)
        {
            double net = 0;
            double snet;

            for (int i = 0; i < x.inputs.Count; i++)
            {
                net += x.inputs[i] * x.weights[i];
            }
            snet = 1 / (1 + Math.Pow(e1, -net));
            return (float)snet;
        }

        public void addSnetTree()
        {
            node NewNode = new node();
            for (int q = 0; q < numberOfHiddenLayers; q++)
            {
                try
                {
                    for (int w = 0; w < tree[q + 1].nodes.Count; w++)
                    {
                        tree[q + 1].nodes[w].value = calcSnet(tree[q + 1].nodes[w]);
                    }
                    try
                    {
                        for (int ww = 0; ww < numberOfNodes[q]; ww++)
                        {
                            for (int a = 0; a < tree[q + 2].nodes.Count; a++)
                            {
                                if (tree[q + 2].nodes[a].inputs.Count < tree[q + 1].nodes.Count)
                                {
                                    tree[q + 2].nodes[a].inputs.Add(tree[q + 1].nodes[ww].value);
                                }
                                else
                                {
                                    tree[q + 2].nodes[a].inputs[ww] = tree[q + 1].nodes[ww].value;
                                }
                            }
                        }
                    }
                    catch (Exception er)
                    {
                        Console.WriteLine("Exception: " + er.Message);
                    }
                }
                catch (Exception e33)
                {
                    Console.WriteLine("Exception: " + e33.Message);
                }

            }
            for (int h = 0; h < 2; h++)
            {
                tree[numberOfHiddenLayers + 1].nodes[h].value = calcSnet(tree[numberOfHiddenLayers + 1].nodes[h]);
            }
        }
        public void splitBySpace(string x)
        {
            layer input = new layer();

            node NewNode = new node();
            string[] allInputs = new string[7];
            allInputs = x.Split(' ');

            for (int j = 0; j < 6; j++)
            {
                NewNode.value = float.Parse(allInputs[j]);

                NewNode.Layernum = 0;
                NewNode.Nodenum = j;
                input.nodes.Add(NewNode);
                NewNode = new node();
            }
            input.Layernum = 0;

            //makes sure here it's tree[0]=input after first time
            if (tree.Count <= 0)
            {
                tree.Add(input);
                treeBest.Add(input);
            }
            else
            {
                tree[0] = input;
            }
            desired = allInputs[6];
            if (desired == "DH")
            {
                desired1 = 0; desired2 = 0;
            }
            else if (desired == "SL")
            {
                desired1 = 0; desired2 = 1;
            }
            else if (desired == "NO")
            {
                desired1 = 1; desired2 = 0;
            }
        }

        public void addOutput()
        {
            layer output = new layer();
            node NewNode = new node();
            int flag2 = 0;
            //number of nodes in each layer
            for (int i = 0; i < 2; i++)
            {

                NewNode.Layernum = numberOfHiddenLayers + 1;
                NewNode.Nodenum = i;
                //A loop for all inputs &weights
                try
                {
                    for (int j = 0; j < tree[NewNode.Layernum - 1].nodes.Count; j++)
                    {
                        //generating weights
                        if (tree.Count < numberOfHiddenLayers + 2)
                        {
                            RandomNumber = (float)rand2.NextDouble();
                            NewNode.weights.Add(RandomNumber);
                            RandomNumber = 0;
                        }
                        else
                        {
                            flag2 = 1;
                        }
                    }
                    if (flag2 == 1)
                    {
                        for (int q = 0; q < tree[numberOfHiddenLayers + 1].nodes[i].weights.Count; q++)
                        {
                            NewNode.weights.Add(tree[numberOfHiddenLayers + 1].nodes[i].weights[q]);
                        }
                        flag2 = 0;
                    }
                    output.nodes.Add(NewNode);
                    NewNode = new node();
                }
                catch (Exception e)
                {
                    Console.WriteLine("Exception: " + e.Message);
                }
            }
            output.Layernum = numberOfHiddenLayers + 1;
            if (tree.Count < numberOfHiddenLayers + 2)
            {
                tree.Add(output);
                treeBest.Add(output);
            }
            else
            {
                try
                {
                    tree[numberOfHiddenLayers + 1] = output;
                }
                catch (Exception er)
                {

                }
            }
        }
        public void addHidden(int a)
        {
            layer hidden = new layer();
            node NewNode = new node();
            int flag = 0;
            //number of nodes in each layer
            for (int i = 0; i < numberOfNodes[a]; i++)
            {
                NewNode.Layernum = a + 1;
                NewNode.Nodenum = i;
                //A loop for all inputs &weights
                for (int j = 0; j < tree[a].nodes.Count; j++)
                {
                    if (a == 0)
                    {
                        NewNode.inputs.Add(tree[a].nodes[j].value);
                    }
                    //generating weights
                    if (tree.Count < numberOfHiddenLayers + 2)
                    {
                        RandomNumber = (float)rand2.NextDouble();
                        NewNode.weights.Add(RandomNumber);
                        RandomNumber = 0;
                    }
                    else
                    {
                        //get weights from old tree
                        flag = 1;
                    }
                }
                if (flag == 1)
                {
                    for (int q = 0; q < tree[a + 1].nodes[i].weights.Count; q++)
                    {
                        NewNode.weights.Add(tree[a + 1].nodes[i].weights[q]);
                    }
                    flag = 0;
                }
                hidden.nodes.Add(NewNode);
                NewNode = new node();
            }
            hidden.Layernum = a + 1;
            if (tree.Count < numberOfHiddenLayers + 2)
            {
                tree.Add(hidden);
                treeBest.Add(hidden);
            }
            else
            {
                try
                {
                    tree[a + 1] = hidden;
                }
                catch (Exception er)
                {

                }
            }
        }
        public void ReadFile()
        {
            String line;
            try
            {
                //Pass the file path and file name to the StreamReader
                StreamReader sr = new StreamReader("samples.txt");

                //Read the first line
                line = sr.ReadLine();

                //Continue to read until the end of file
                while (line != null)
                {
                    LineSample.Add(line);
                    //Read the next line
                    line = sr.ReadLine();
                }

                //close file
                sr.Close();
                Console.ReadLine();
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
        }
        public void WriteFile2(string txt)
        {
            try
            {
                sw.WriteLine(txt);
            }
            catch (Exception e)
            {
                Console.WriteLine("Exception: " + e.Message);
            }
        }
       
        public Form1()
        {
            InitializeComponent();
            ReadFile();
            
        }

        public void EnterHidden()
        {
            if (numberOfEnterClicks == 0 && textBox1.Text != "")
            {
                numberOfHiddenLayers = Int32.Parse(textBox1.Text);
                numberOfEnterClicks++;
            }
            if (numberOfEnterClicks > 0 && numberOfEnterClicks - 1 <= numberOfHiddenLayers && textBox1.Text != "")
            {
                if (numberOfEnterClicks >= 2)
                {
                    numberOfNodes.Add(Int32.Parse(textBox1.Text));
                }
                label1.Text = "Enter Number of Nodes for hidden layer number :" + "" + numberOfEnterClicks;
                numberOfEnterClicks++;
                textBox1.Text = "";
            }
            if (numberOfHiddenLayers == numberOfEnterClicks - 2)
            {
                //button1.Text = "Train";
                label1.Text = "Enter Mean Square Error :";
                otherInputs = 1; numberOfEnterClicks = 0;
            }
        }

        public void EnterOtherInputs()
        {
            if (numberOfEnterClicks == 0 && textBox1.Text != "")
            {
                numberOfEnterClicks++;
                mse = float.Parse(textBox1.Text);
                textBox1.Text = "";
                label1.Text = "Enter Learning Rate :";
            }
            else if (numberOfEnterClicks == 1 && textBox1.Text != "")
            {
                numberOfEnterClicks++;
                learnRate = float.Parse(textBox1.Text);
                textBox1.Text = "";
                label1.Text = "Enter Momenteum Factor :";
            }
            else if (numberOfEnterClicks == 2 && textBox1.Text != "")
            {
                numberOfEnterClicks++; otherInputs = 2;
                momentum = float.Parse(textBox1.Text);
                textBox1.Text = "";
                textBox1.Enabled = false;
                label1.Text = "Press Train to start !";
                button1.Text = "Train";
            }
        }
        private void button1_Click(object sender, EventArgs e)
        {
            if (otherInputs == 0) { EnterHidden(); }
            else if (otherInputs == 1) { EnterOtherInputs(); }
            else if (otherInputs == 2)
            {
                //we make a loop here for all the tree
                label1.Text = "Loading...";
                while (true)
                {
                    //breaks when howManyGood=LineSample.count*70/100;

                    splitBySpace(LineSample[ii]); //adds the input
                    for (int a = 0; a < numberOfHiddenLayers; a++)
                    {
                        addHidden(a);
                    }
                    addOutput();

                    addSnetTree(); //can also be used as update snet
                    try
                    {
                        for (int b = 0; b < tree[numberOfHiddenLayers + 1].nodes.Count; b++)
                        {
                            Errorsqd += calcErrorSqd(tree[numberOfHiddenLayers + 1].nodes[b]);
                        }
                        Errorsqd /= 2;
                    }
                    catch (Exception e3)
                    {
                        Console.WriteLine("Exception: " + e3.Message);
                    }
                    if (Errorsqd <= mse)
                    {
                        //WriteFile("Data is ok");
                        howManyGood++;
                        if (howManyGood > bestHowManyGood)
                        {
                            bestHowManyGood = howManyGood;
                            for (int ij = 0; ij < tree.Count; ij++)
                            {
                                for (int fg = 0; fg < tree[ij].nodes.Count; fg++)
                                {
                                    for (int lg = 0; lg < tree[ij].nodes[fg].weights.Count; lg++)
                                    {
                                        treeBest[ij].nodes[fg].weights[lg] = tree[ij].nodes[fg].weights[lg];
                                        treeBest[ij].nodes[fg].inputs[lg] = tree[ij].nodes[fg].inputs[lg];
                                    }
                                }
                            }
                        }
                        if (howManyGood >= LineSample.Count * 70 / 100)
                        {
                            label1.Text = "Done";
                            button1.Text = "Test";
                            otherInputs = 5;
                            //sw.Close();
                            break;
                        }
                    }
                    else
                    {
                        //label1.Text = "Data is BAD :(";
                        for (int ab = 0; ab < tree[numberOfHiddenLayers + 1].nodes.Count; ab++)
                        {
                            OutputError = outputError(tree[numberOfHiddenLayers + 1].nodes[ab]);
                            tree[numberOfHiddenLayers + 1].nodes[ab].error = OutputError;
                        }
                        for (int f = numberOfHiddenLayers; f >= 1; f--)
                        {
                            for (int z = 0; z < tree[f].nodes.Count; z++)
                            {
                                HiddenError = hidError(tree[f].nodes[z]);
                                tree[f].nodes[z].error = HiddenError;
                            }
                        }
                        //updating weights
                        //label1.Text = "";
                        for (int ai = 1; ai < tree.Count; ai++)
                        {
                            for (int da = 0; da < tree[ai].nodes.Count; da++)
                            {
                                //label1.Text += tree[ai].nodes[da].inputs.Count+",";
                                tree[ai].nodes[da] = updateWeights(tree[ai].nodes[da]);
                            }
                        }
                        itterations++;
                        //break;
                        howManyGood = 1;
                    }
                    ii++;
                    Errorsqd = 0;

                    if (ii >= LineSample.Count * 70 / 100)
                    {
                        ii = 0;
                    }
                    //get most good weights instead
                    if (itterations >= LineSample.Count * 200)
                    {
                        for (int ija = 0; ija < tree.Count; ija++)
                        {
                            for (int fga = 0; fga < tree[ija].nodes.Count; fga++)
                            {
                                for (int lga = 0; lga < tree[ija].nodes[fga].weights.Count; lga++)
                                {
                                    tree[ija].nodes[fga].weights[lga] = treeBest[ija].nodes[fga].weights[lga];
                                    tree[ija].nodes[fga].inputs[lga] = treeBest[ija].nodes[fga].inputs[lga];
                                }
                            }
                        }
                        label1.Text = "Done !";
                        button1.Text = "Test";
                        otherInputs = 5;
                        break;
                    }
                }
            }
            else if (otherInputs == 5)
            {
                Errorsqd = 0;
                int rob3;
                if ((LineSample.Count * 70 / 100) % 2 == 1)
                {
                    rob3 = (LineSample.Count * 70 / 100) + 1;
                }
                else
                {
                    rob3 = (LineSample.Count * 70 / 100);
                }
                for (int s = rob3; s < LineSample.Count; s++)
                {
                    splitBySpace(LineSample[s]); //adds the input
                    for (int a = 0; a < numberOfHiddenLayers; a++)
                    {
                        addHidden(a);
                    }
                    addOutput();

                    addSnetTree(); //can also be used as update snet
                    try
                    {
                        for (int b = 0; b < tree[numberOfHiddenLayers + 1].nodes.Count; b++)
                        {
                            Errorsqd += calcErrorSqd(tree[numberOfHiddenLayers + 1].nodes[b]);
                        }
                        Errorsqd /= 2;
                    }
                    catch (Exception e3)
                    {
                        Console.WriteLine("Exception: " + e3.Message);
                    }
                    if (Errorsqd <= mse)
                    {
                        howManyAccurate++;
                    }
                    //MessageBox.Show("" + Errorsqd);
                    Errorsqd = 0;
                }
                float Accuracy = (howManyAccurate / (LineSample.Count * 30 / 100)) * 100;

                label1.Text = "Accuracy = " + Accuracy + "%";

                otherInputs = 6;
                //howManyAccurate = 0;
                string finaltxt = "";
                WriteFile2("NOTE : NODES ARE 0 BASE AND LAYERS ARE 1 BASE");
                WriteFile2("\n\r\n\r");
                for (int l = 1; l < tree.Count; l++)
                {
                    for (int n = 0; n < tree[l].nodes.Count; n++)
                    {
                        for (int w = 0; w < tree[l].nodes[n].weights.Count; w++)
                        {
                            finaltxt += tree[l].nodes[n].weights[w] + " , ";
                        }
                        int v = n + 1;
                        int v2 = l - 1;
                        WriteFile2("Weights from Layer " + v2 + " to Layer" + l + " Node " + v + " = ");
                        WriteFile2(finaltxt);
                        WriteFile2("");
                        finaltxt = "";
                    }
                }
                sw.Close();
                TimeSpan ts = DateTime.Now - dt;
                MessageBox.Show("Time = " + ts.TotalMilliseconds.ToString());
            }

        }
    }
}