using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Media;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LauncherApp
{
    public partial class Form1 : Form
    {
        sharpSettingsReaderWriter settingsFileHandler=new sharpSettingsReaderWriter();
        sharpSettingsDat settings;

        int measurementMaxCharsTarget=100;
        int measurementIncChars=1;
        int parallelInvocs=4;
        int measurementRuns=1;
        bool isParallel = false;
        bool isQuadruped = false;
        bool isConsole = true;
        bool runAllConfigs = false;
        int characterAmount = 1;

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
            progressBar_configPrcnt.Value = 0;
            if (!runAllConfigs)
            {
                runCurrentConfig();
                progressBar_configPrcnt.Value = 4;
            }
            else
            {
                bool oldIsQuadruped = isQuadruped;
                bool oldIsParallel = isParallel;
                int oldParInvoc = parallelInvocs;
                int paraMax = parallelInvocs;
                // biped serial
                isQuadruped = false;
                isParallel = false; 
                progressBar_configPrcnt.Value = 25;
                progressBar_configPrcnt.Update();
                runCurrentConfig();
                // quadruped serial
                isQuadruped = true;
                isParallel = false;
                progressBar_configPrcnt.Value = 50;
                progressBar_configPrcnt.Update();
                runCurrentConfig();

                // biped parallel
                isQuadruped = false;
                isParallel = true;
                for (int i = 2; i <= paraMax; i++)
                {
                    parallelInvocs = i;
                    progressBar_configPrcnt.Value = 50 + (int)(((float)i / (float)paraMax) * 25.0f);
                    progressBar_configPrcnt.Update();
                    runCurrentConfig();
                }

                // quadruped parallel
                isQuadruped = true;
                isParallel = true;
                for (int i = 2; i <= paraMax; i++)
                {
                    parallelInvocs = i;                
                    progressBar_configPrcnt.Value = 75 + (int)(((float)i/(float)paraMax)*25.0f);
                    progressBar_configPrcnt.Update();
                    runCurrentConfig();
                }
                // done, reset
                isQuadruped = oldIsQuadruped;
                isParallel = oldIsParallel;
                parallelInvocs = oldParInvoc;
            }
        }


        void runCurrentConfig()
        {
            measurementStatusListBox.Items.Clear();
            progressBar_characterPrcnt.Value = 0;

            measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Running Experiment...");

            int stepsz = measurementIncChars;

            sharpSettingsDat origsettings = settings;

            // RunInstallerAttribute experiment
            settings.m_parallel_invocs = parallelInvocs;
            settings.m_simMode = "m";
            settings.m_appMode = isConsole ? "c" : "g";
            string exetextversion = isParallel ? "parallel" : "serial";
            settings.m_execMode = isParallel ? "p" : "s";
            string exetextpod = isQuadruped ? "quadruped" : "biped";
            settings.m_pod = isQuadruped ? "q" : "b";

            settings.m_measurementRuns = measurementRuns;

            if (measurementMaxCharsTarget > 1)
                measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Experiment for " + exetextpod + " is " + exetextversion + ", with " + measurementRuns + " runs for each measurement.");
            else
                measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Experiment for " + exetextpod + " has 1 " + exetextversion + " scenario, with " + measurementRuns + " runs.");
            if (isParallel) measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Parallel loop invocations will be" + parallelInvocs + ".");
            int mcount = 0;
            for (int i = 0; i <= measurementMaxCharsTarget; i += stepsz)
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
                progressBar_characterPrcnt.Value = (int)(100.0f * ((float)i) / ((float)measurementMaxCharsTarget));
                progressBar_characterPrcnt.Update();
                progressBar_configPrcnt.Update();
                Application.DoEvents();
            }
            measurementStatusListBox.TopIndex = measurementStatusListBox.Items.Add("Experiment finished!");

            string exePathPrefix = Application.StartupPath;
            SoundPlayer simpleSound = new SoundPlayer(exePathPrefix + "/../Music_Box.wav");
            simpleSound.Play();
            progressBar_characterPrcnt.Value = 100;

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

        private void biped_radioButton_CheckedChanged(object sender, EventArgs e)
        {
            if (biped_radioButton.Checked) isQuadruped = false;
        }

        private void quadruped_radioButton_CheckedChanged(object sender, EventArgs e)
        {
            if (quadruped_radioButton.Checked) isQuadruped = true;
        }

        private void checkBox1_CheckedChanged(object sender, EventArgs e)
        {
            isConsole = checkBox1.Checked;
        }

        private void button1_Click(object sender, EventArgs e)
        {
            sharpSettingsDat origsettings = settings;

            // RunInstallerAttribute experiment
            settings.m_parallel_invocs = parallelInvocs;
            settings.m_simMode = "r";
            settings.m_appMode = isConsole ? "c" : "g";
            settings.m_execMode = isParallel ? "p" : "s";
            settings.m_pod = isQuadruped ? "q" : "b";
            settings.m_charcount_serial = characterAmount;
            //
            settingsFileHandler.writeSettings(settings);

            Process winapp = new Process();
            winapp.StartInfo.FileName = "WinApp_Release.exe";
            //winapp.StartInfo.Arguments = "/r:System.dll /out:sample.exe stdstr.cs";
            winapp.StartInfo.UseShellExecute = false;
            //winapp.StartInfo.RedirectStandardOutput = true;
            winapp.Start();
            winapp.WaitForExit();
            settings = origsettings;
            settingsFileHandler.writeSettings(settings);
        }

        private void textBox3_TextChanged(object sender, EventArgs e)
        {
            try
            {
                // Convert the text to a Double and determine if it is a negative number. 
                int amount = int.Parse(textBox3.Text);
                if (amount < 1) { amount = 1; textBox3.Text = "1"; }
                characterAmount = amount;
            }
            catch
            {
                // If there is an error, display the text using the system colors.
                MaxChars.ForeColor = SystemColors.ControlText;
            }
        }

        private void textBox2_TextChanged(object sender, EventArgs e)
        {
            try
            {
                // Convert the text to a Double and determine if it is a negative number. 
                int amount = int.Parse(textBox2.Text);
                if (amount < 1) { amount = 1; textBox2.Text = "1"; }
                parallelInvocs = amount;
            }
            catch
            {
                // If there is an error, display the text using the system colors.
                parInvocSz.ForeColor = SystemColors.ControlText;
            }
        }

        private void label10_Click(object sender, EventArgs e)
        {

        }

        private void checkBox2_CheckedChanged(object sender, EventArgs e)
        {
            runAllConfigs = checkBox2.Checked;
        }

        private void progressBar_characterPrcnt_Click(object sender, EventArgs e)
        {

        }

        private void progressBar_configPrcnt_Click(object sender, EventArgs e)
        {

        }

    }
}
