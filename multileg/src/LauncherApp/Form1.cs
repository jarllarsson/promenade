using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LauncherApp
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {

        }

        private void button1_Click(object sender, EventArgs e)
        {
            Process winapp = new Process();
            winapp.StartInfo.FileName = "WinApp_Release.exe";
            //winapp.StartInfo.Arguments = "/r:System.dll /out:sample.exe stdstr.cs";
            winapp.StartInfo.UseShellExecute = false;
            //winapp.StartInfo.RedirectStandardOutput = true;
            winapp.Start();    
            //Console.WriteLine(winapp.StandardOutput.ReadToEnd());

            winapp.WaitForExit();
            Console.WriteLine("DONE");
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {

        }
    }
}
