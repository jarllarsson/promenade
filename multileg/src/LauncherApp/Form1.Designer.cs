namespace LauncherApp
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Form1));
            this.pictureBox1 = new System.Windows.Forms.PictureBox();
            this.runMeasurementBtn = new System.Windows.Forms.Button();
            this.serialRadioBtn = new System.Windows.Forms.RadioButton();
            this.parallelRadioBtn = new System.Windows.Forms.RadioButton();
            this.MaxChars = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.charIncVal = new System.Windows.Forms.TextBox();
            this.tabControl1 = new System.Windows.Forms.TabControl();
            this.tabPage1 = new System.Windows.Forms.TabPage();
            this.progressBar_configPrcnt = new System.Windows.Forms.ProgressBar();
            this.progressBar_characterPrcnt = new System.Windows.Forms.ProgressBar();
            this.label10 = new System.Windows.Forms.Label();
            this.checkBox2 = new System.Windows.Forms.CheckBox();
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.measurementStatusListBox = new System.Windows.Forms.ListBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.parInvocSz = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            this.label7 = new System.Windows.Forms.Label();
            this.textBox2 = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.textBox3 = new System.Windows.Forms.TextBox();
            this.label9 = new System.Windows.Forms.Label();
            this.button1 = new System.Windows.Forms.Button();
            this.tabPage3 = new System.Windows.Forms.TabPage();
            this.biped_radioButton = new System.Windows.Forms.RadioButton();
            this.quadruped_radioButton = new System.Windows.Forms.RadioButton();
            this.panel1 = new System.Windows.Forms.Panel();
            this.panel2 = new System.Windows.Forms.Panel();
            this.checkBox1 = new System.Windows.Forms.CheckBox();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
            this.tabPage2.SuspendLayout();
            this.panel1.SuspendLayout();
            this.panel2.SuspendLayout();
            this.SuspendLayout();
            // 
            // pictureBox1
            // 
            this.pictureBox1.Image = ((System.Drawing.Image)(resources.GetObject("pictureBox1.Image")));
            this.pictureBox1.Location = new System.Drawing.Point(96, 12);
            this.pictureBox1.Name = "pictureBox1";
            this.pictureBox1.Size = new System.Drawing.Size(137, 127);
            this.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.StretchImage;
            this.pictureBox1.TabIndex = 0;
            this.pictureBox1.TabStop = false;
            this.pictureBox1.Click += new System.EventHandler(this.pictureBox1_Click);
            // 
            // runMeasurementBtn
            // 
            this.runMeasurementBtn.Location = new System.Drawing.Point(18, 205);
            this.runMeasurementBtn.Name = "runMeasurementBtn";
            this.runMeasurementBtn.Size = new System.Drawing.Size(268, 23);
            this.runMeasurementBtn.TabIndex = 1;
            this.runMeasurementBtn.Text = "Run Measurement";
            this.runMeasurementBtn.UseVisualStyleBackColor = true;
            this.runMeasurementBtn.Click += new System.EventHandler(this.runMeasurement_Click);
            // 
            // serialRadioBtn
            // 
            this.serialRadioBtn.AutoSize = true;
            this.serialRadioBtn.Checked = true;
            this.serialRadioBtn.Location = new System.Drawing.Point(3, 3);
            this.serialRadioBtn.Name = "serialRadioBtn";
            this.serialRadioBtn.Size = new System.Drawing.Size(51, 17);
            this.serialRadioBtn.TabIndex = 2;
            this.serialRadioBtn.TabStop = true;
            this.serialRadioBtn.Text = "Serial";
            this.serialRadioBtn.UseVisualStyleBackColor = true;
            this.serialRadioBtn.CheckedChanged += new System.EventHandler(this.radioButton1_CheckedChanged);
            // 
            // parallelRadioBtn
            // 
            this.parallelRadioBtn.AutoSize = true;
            this.parallelRadioBtn.Location = new System.Drawing.Point(60, 3);
            this.parallelRadioBtn.Name = "parallelRadioBtn";
            this.parallelRadioBtn.Size = new System.Drawing.Size(59, 17);
            this.parallelRadioBtn.TabIndex = 3;
            this.parallelRadioBtn.TabStop = true;
            this.parallelRadioBtn.Text = "Parallel";
            this.parallelRadioBtn.UseVisualStyleBackColor = true;
            this.parallelRadioBtn.CheckedChanged += new System.EventHandler(this.parallelRadioBtn_CheckedChanged);
            // 
            // MaxChars
            // 
            this.MaxChars.Location = new System.Drawing.Point(6, 23);
            this.MaxChars.Name = "MaxChars";
            this.MaxChars.Size = new System.Drawing.Size(100, 20);
            this.MaxChars.TabIndex = 4;
            this.MaxChars.Text = "1";
            this.MaxChars.TextChanged += new System.EventHandler(this.MaxChars_TextChanged);
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(3, 7);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(140, 13);
            this.label1.TabIndex = 5;
            this.label1.Text = "Run to this amount of chars:";
            this.label1.Click += new System.EventHandler(this.label1_Click);
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(3, 57);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(159, 13);
            this.label2.TabIndex = 7;
            this.label2.Text = "Each experiment increases with:";
            this.label2.Click += new System.EventHandler(this.label2_Click);
            // 
            // charIncVal
            // 
            this.charIncVal.Location = new System.Drawing.Point(6, 73);
            this.charIncVal.Name = "charIncVal";
            this.charIncVal.Size = new System.Drawing.Size(100, 20);
            this.charIncVal.TabIndex = 6;
            this.charIncVal.Text = "1";
            this.charIncVal.TextChanged += new System.EventHandler(this.IncChars_TextChanged);
            // 
            // tabControl1
            // 
            this.tabControl1.Controls.Add(this.tabPage1);
            this.tabControl1.Controls.Add(this.tabPage2);
            this.tabControl1.Controls.Add(this.tabPage3);
            this.tabControl1.Location = new System.Drawing.Point(12, 238);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(317, 425);
            this.tabControl1.TabIndex = 8;
            // 
            // tabPage1
            // 
            this.tabPage1.Controls.Add(this.progressBar_configPrcnt);
            this.tabPage1.Controls.Add(this.progressBar_characterPrcnt);
            this.tabPage1.Controls.Add(this.label10);
            this.tabPage1.Controls.Add(this.checkBox2);
            this.tabPage1.Controls.Add(this.textBox1);
            this.tabPage1.Controls.Add(this.label6);
            this.tabPage1.Controls.Add(this.measurementStatusListBox);
            this.tabPage1.Controls.Add(this.label5);
            this.tabPage1.Controls.Add(this.label4);
            this.tabPage1.Controls.Add(this.parInvocSz);
            this.tabPage1.Controls.Add(this.label3);
            this.tabPage1.Controls.Add(this.charIncVal);
            this.tabPage1.Controls.Add(this.label2);
            this.tabPage1.Controls.Add(this.runMeasurementBtn);
            this.tabPage1.Controls.Add(this.MaxChars);
            this.tabPage1.Controls.Add(this.label1);
            this.tabPage1.Location = new System.Drawing.Point(4, 22);
            this.tabPage1.Name = "tabPage1";
            this.tabPage1.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage1.Size = new System.Drawing.Size(309, 399);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Measurement";
            this.tabPage1.UseVisualStyleBackColor = true;
            // 
            // progressBar_configPrcnt
            // 
            this.progressBar_configPrcnt.Location = new System.Drawing.Point(7, 370);
            this.progressBar_configPrcnt.Name = "progressBar_configPrcnt";
            this.progressBar_configPrcnt.Size = new System.Drawing.Size(296, 23);
            this.progressBar_configPrcnt.Step = 25;
            this.progressBar_configPrcnt.TabIndex = 18;
            this.progressBar_configPrcnt.Click += new System.EventHandler(this.progressBar_configPrcnt_Click);
            // 
            // progressBar_characterPrcnt
            // 
            this.progressBar_characterPrcnt.Location = new System.Drawing.Point(5, 344);
            this.progressBar_characterPrcnt.Name = "progressBar_characterPrcnt";
            this.progressBar_characterPrcnt.Size = new System.Drawing.Size(296, 10);
            this.progressBar_characterPrcnt.TabIndex = 17;
            this.progressBar_characterPrcnt.Click += new System.EventHandler(this.progressBar_characterPrcnt_Click);
            // 
            // label10
            // 
            this.label10.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Italic, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label10.Location = new System.Drawing.Point(187, 132);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(99, 40);
            this.label10.TabIndex = 16;
            this.label10.Text = "(read Parallel Invocations as 2 to value)";
            this.label10.Click += new System.EventHandler(this.label10_Click);
            // 
            // checkBox2
            // 
            this.checkBox2.AutoSize = true;
            this.checkBox2.Location = new System.Drawing.Point(190, 109);
            this.checkBox2.Name = "checkBox2";
            this.checkBox2.Size = new System.Drawing.Size(96, 17);
            this.checkBox2.TabIndex = 15;
            this.checkBox2.Text = "Run all configs";
            this.checkBox2.UseVisualStyleBackColor = true;
            this.checkBox2.CheckedChanged += new System.EventHandler(this.checkBox2_CheckedChanged);
            // 
            // textBox1
            // 
            this.textBox1.Location = new System.Drawing.Point(9, 175);
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(100, 20);
            this.textBox1.TabIndex = 13;
            this.textBox1.TextChanged += new System.EventHandler(this.textBox1_TextChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(6, 159);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(94, 13);
            this.label6.TabIndex = 14;
            this.label6.Text = "Sample Size (runs)";
            this.label6.Click += new System.EventHandler(this.label6_Click);
            // 
            // measurementStatusListBox
            // 
            this.measurementStatusListBox.FormattingEnabled = true;
            this.measurementStatusListBox.Items.AddRange(new object[] {
            "test1",
            "test2",
            "test3",
            "test4"});
            this.measurementStatusListBox.Location = new System.Drawing.Point(7, 230);
            this.measurementStatusListBox.Name = "measurementStatusListBox";
            this.measurementStatusListBox.Size = new System.Drawing.Size(296, 108);
            this.measurementStatusListBox.TabIndex = 12;
            this.measurementStatusListBox.SelectedIndexChanged += new System.EventHandler(this.measurementStatusListBox_SelectedIndexChanged);
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(106, 30);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(57, 13);
            this.label5.TabIndex = 11;
            this.label5.Text = "characters";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(106, 80);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(57, 13);
            this.label4.TabIndex = 10;
            this.label4.Text = "characters";
            // 
            // parInvocSz
            // 
            this.parInvocSz.Location = new System.Drawing.Point(6, 125);
            this.parInvocSz.Name = "parInvocSz";
            this.parInvocSz.Size = new System.Drawing.Size(100, 20);
            this.parInvocSz.TabIndex = 8;
            this.parInvocSz.TextChanged += new System.EventHandler(this.parInvocSz_TextChanged);
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(3, 109);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(133, 13);
            this.label3.TabIndex = 9;
            this.label3.Text = "Parallel Invocations (if any)";
            // 
            // tabPage2
            // 
            this.tabPage2.Controls.Add(this.label7);
            this.tabPage2.Controls.Add(this.textBox2);
            this.tabPage2.Controls.Add(this.label8);
            this.tabPage2.Controls.Add(this.textBox3);
            this.tabPage2.Controls.Add(this.label9);
            this.tabPage2.Controls.Add(this.button1);
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(309, 399);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "Run";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(106, 26);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(57, 13);
            this.label7.TabIndex = 16;
            this.label7.Text = "characters";
            // 
            // textBox2
            // 
            this.textBox2.Location = new System.Drawing.Point(6, 68);
            this.textBox2.Name = "textBox2";
            this.textBox2.Size = new System.Drawing.Size(100, 20);
            this.textBox2.TabIndex = 14;
            this.textBox2.TextChanged += new System.EventHandler(this.textBox2_TextChanged);
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(3, 52);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(133, 13);
            this.label8.TabIndex = 15;
            this.label8.Text = "Parallel Invocations (if any)";
            // 
            // textBox3
            // 
            this.textBox3.Location = new System.Drawing.Point(6, 19);
            this.textBox3.Name = "textBox3";
            this.textBox3.Size = new System.Drawing.Size(100, 20);
            this.textBox3.TabIndex = 12;
            this.textBox3.Text = "1";
            this.textBox3.TextChanged += new System.EventHandler(this.textBox3_TextChanged);
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(3, 3);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(87, 13);
            this.label9.TabIndex = 13;
            this.label9.Text = "Amount of chars:";
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(20, 163);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(268, 23);
            this.button1.TabIndex = 2;
            this.button1.Text = "Run Simulation";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.button1_Click);
            // 
            // tabPage3
            // 
            this.tabPage3.Location = new System.Drawing.Point(4, 22);
            this.tabPage3.Name = "tabPage3";
            this.tabPage3.Size = new System.Drawing.Size(309, 399);
            this.tabPage3.TabIndex = 2;
            this.tabPage3.Text = "Optimize";
            this.tabPage3.UseVisualStyleBackColor = true;
            // 
            // biped_radioButton
            // 
            this.biped_radioButton.AutoSize = true;
            this.biped_radioButton.Checked = true;
            this.biped_radioButton.Location = new System.Drawing.Point(4, 3);
            this.biped_radioButton.Name = "biped_radioButton";
            this.biped_radioButton.Size = new System.Drawing.Size(52, 17);
            this.biped_radioButton.TabIndex = 9;
            this.biped_radioButton.TabStop = true;
            this.biped_radioButton.Text = "Biped";
            this.biped_radioButton.UseVisualStyleBackColor = true;
            this.biped_radioButton.CheckedChanged += new System.EventHandler(this.biped_radioButton_CheckedChanged);
            // 
            // quadruped_radioButton
            // 
            this.quadruped_radioButton.AutoSize = true;
            this.quadruped_radioButton.Location = new System.Drawing.Point(61, 3);
            this.quadruped_radioButton.Name = "quadruped_radioButton";
            this.quadruped_radioButton.Size = new System.Drawing.Size(78, 17);
            this.quadruped_radioButton.TabIndex = 10;
            this.quadruped_radioButton.TabStop = true;
            this.quadruped_radioButton.Text = "Quadruped";
            this.quadruped_radioButton.UseVisualStyleBackColor = true;
            this.quadruped_radioButton.CheckedChanged += new System.EventHandler(this.quadruped_radioButton_CheckedChanged);
            // 
            // panel1
            // 
            this.panel1.Controls.Add(this.biped_radioButton);
            this.panel1.Controls.Add(this.quadruped_radioButton);
            this.panel1.Location = new System.Drawing.Point(12, 166);
            this.panel1.Name = "panel1";
            this.panel1.Size = new System.Drawing.Size(147, 27);
            this.panel1.TabIndex = 11;
            // 
            // panel2
            // 
            this.panel2.Controls.Add(this.serialRadioBtn);
            this.panel2.Controls.Add(this.parallelRadioBtn);
            this.panel2.Location = new System.Drawing.Point(12, 204);
            this.panel2.Name = "panel2";
            this.panel2.Size = new System.Drawing.Size(126, 28);
            this.panel2.TabIndex = 12;
            // 
            // checkBox1
            // 
            this.checkBox1.AutoSize = true;
            this.checkBox1.Checked = true;
            this.checkBox1.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBox1.Location = new System.Drawing.Point(185, 166);
            this.checkBox1.Name = "checkBox1";
            this.checkBox1.Size = new System.Drawing.Size(93, 17);
            this.checkBox1.TabIndex = 15;
            this.checkBox1.Text = "Console mode";
            this.checkBox1.UseVisualStyleBackColor = true;
            this.checkBox1.CheckedChanged += new System.EventHandler(this.checkBox1_CheckedChanged);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(341, 689);
            this.Controls.Add(this.checkBox1);
            this.Controls.Add(this.panel2);
            this.Controls.Add(this.panel1);
            this.Controls.Add(this.tabControl1);
            this.Controls.Add(this.pictureBox1);
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Name = "Form1";
            this.Text = "Launch";
            this.Load += new System.EventHandler(this.Form1_Load);
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).EndInit();
            this.tabControl1.ResumeLayout(false);
            this.tabPage1.ResumeLayout(false);
            this.tabPage1.PerformLayout();
            this.tabPage2.ResumeLayout(false);
            this.tabPage2.PerformLayout();
            this.panel1.ResumeLayout(false);
            this.panel1.PerformLayout();
            this.panel2.ResumeLayout(false);
            this.panel2.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.PictureBox pictureBox1;
        private System.Windows.Forms.Button runMeasurementBtn;
        private System.Windows.Forms.RadioButton serialRadioBtn;
        private System.Windows.Forms.RadioButton parallelRadioBtn;
        private System.Windows.Forms.TextBox MaxChars;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox charIncVal;
        private System.Windows.Forms.TabControl tabControl1;
        private System.Windows.Forms.TabPage tabPage1;
        private System.Windows.Forms.TabPage tabPage2;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox parInvocSz;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.ListBox measurementStatusListBox;
        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.RadioButton biped_radioButton;
        private System.Windows.Forms.RadioButton quadruped_radioButton;
        private System.Windows.Forms.Panel panel1;
        private System.Windows.Forms.Panel panel2;
        private System.Windows.Forms.CheckBox checkBox1;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.TabPage tabPage3;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox textBox2;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox textBox3;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.CheckBox checkBox2;
        private System.Windows.Forms.ProgressBar progressBar_configPrcnt;
        private System.Windows.Forms.ProgressBar progressBar_characterPrcnt;
    }
}

