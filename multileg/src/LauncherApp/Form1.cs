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
        sharpSettingsReaderWriter settingsFileHandler=new sharpSettingsReaderWriter();
        sharpSettingsDat settings;

        int measurementMaxCharsTarget=1;
        int measurementIncChars=1;
        int parallelInvocs=4;
        int measurementRuns=1;
        bool isParallel = false;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            settingsFileHandler.loadSettings(ref settings);
            textBox1.Text = settings.m_measurementRuns.ToString();
            parInvocSz.Text = settings.m_parallel_invocs.ToString();
        }

        private void runMeasurement_Click(object sender, EventArgs e)
        {
            measurementStatusListBox.Items.Clear();

            measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Running Experiment...");

            int stepsz=measurementIncChars;

            sharpSettingsDat origsettings = settings;

            // RunInstallerAttribute experiment
            settings.m_parallel_invocs = parallelInvocs;
            settings.m_simMode = "m";
            settings.m_appMode = "c";
            string exetextversion = isParallel ? "parallel" : "serial";
            settings.m_execMode = isParallel?"p":"s";
            settings.m_measurementRuns = measurementRuns;

            if (measurementMaxCharsTarget>1)
                measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Experiment is " + exetextversion + ", with " + measurementRuns + " runs for each measurement.");
            else
                measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Experiment has 1 " + exetextversion + " scenario, with " + measurementRuns + " runs.");
            if (isParallel) measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Parallel loop invocations will be" + parallelInvocs + ".");
            int mcount = 0;
            for (int i=0;i<=measurementMaxCharsTarget;i+=stepsz)
            {
                mcount++;
                int charCount = i;
                if (charCount == 0)
                {
                    charCount = 1;
                    if (stepsz == 1) i++; // make next 2 if step size is only 1
                    measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Executing Control with 1 character");
                }
                else
                {
                    measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Executing Measurement(" + mcount + ") with " + charCount + " characters");
                }
                // write settings
                settings.m_charcount_serial = charCount;
                settingsFileHandler.writeSettings(settings);
                //
                Process winapp = new Process();
                winapp.StartInfo.FileName = "WinApp_Release.exe";
                //winapp.StartInfo.Arguments = "/r:System.dll /out:sample.exe stdstr.cs";
                winapp.StartInfo.UseShellExecute = false;
                //winapp.StartInfo.RedirectStandardOutput = true;
                winapp.Start();
                //Console.WriteLine(winapp.StandardOutput.ReadToEnd());
                winapp.WaitForExit();
            }
            measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Experiment finished!");

            settings = origsettings;
            settingsFileHandler.writeSettings(settings);
        }

        private void pictureBox1_Click(object sender, EventArgs e)
        {

        }

        private void radioButton1_CheckedChanged(object sender, EventArgs e)
        {
            if (serialRadioBtn.Checked) isParallel = false;
        }


        private void parallelRadioBtn_CheckedChanged(object sender, EventArgs e)
        {
            if (parallelRadioBtn.Checked) isParallel = true;
        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void IncChars_TextChanged(object sender, EventArgs e)
        {
            try
            {
                // Convert the text to a Double and determine if it is a negative number. 
                int amount = int.Parse(charIncVal.Text);
                if (amount < 1) { amount = 1; charIncVal.Text = "1"; }
               measurementIncChars = amount;
            }
            catch
            {
                // If there is an error, display the text using the system colors.
                charIncVal.ForeColor = SystemColors.ControlText;
            }
        }

        private void MaxChars_TextChanged(object sender, EventArgs e)
        {
            try
            {
               // Convert the text to a Double and determine if it is a negative number. 
               int amount=int.Parse(MaxChars.Text);
               if (amount < 1) { amount = 1; MaxChars.Text = "1"; }
               measurementMaxCharsTarget = amount;
            }  
            catch
            {
               // If there is an error, display the text using the system colors.
               MaxChars.ForeColor = SystemColors.ControlText;
            }
        }

        private void parInvocSz_TextChanged(object sender, EventArgs e)
        {
            try
            {
                // Convert the text to a Double and determine if it is a negative number. 
                int amount = int.Parse(parInvocSz.Text);
                if (amount < 1) { amount = 1; parInvocSz.Text = "1";}
                parallelInvocs = amount;
            }
            catch
            {
                // If there is an error, display the text using the system colors.
                parInvocSz.ForeColor = SystemColors.ControlText;
            }
        }

        private void measurementStatusListBox_SelectedIndexChanged(object sender, EventArgs e)
        {
            
        }

        private void label6_Click(object sender, EventArgs e)
        {

        }

        private void textBox1_TextChanged(object sender, EventArgs e)
        {
            try
            {
                // Convert the text to a Double and determine if it is a negative number. 
                int amount = int.Parse(textBox1.Text);
                if (amount < 1) { amount = 1; textBox1.Text = "1";}
                measurementRuns = amount;
            }
            catch
            {
                // If there is an error, display the text using the system colors.
                textBox1.ForeColor = SystemColors.ControlText;
            }
        }


    }
}
