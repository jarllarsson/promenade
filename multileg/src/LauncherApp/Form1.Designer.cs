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
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.measurementStatusListBox = new System.Windows.Forms.ListBox();
            this.label5 = new System.Windows.Forms.Label();
            this.label4 = new System.Windows.Forms.Label();
            this.parInvocSz = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.tabPage2 = new System.Windows.Forms.TabPage();
            ((System.ComponentModel.ISupportInitialize)(this.pictureBox1)).BeginInit();
            this.tabControl1.SuspendLayout();
            this.tabPage1.SuspendLayout();
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
            this.serialRadioBtn.Location = new System.Drawing.Point(15, 215);
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
            this.parallelRadioBtn.Location = new System.Drawing.Point(73, 215);
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
            this.tabControl1.Location = new System.Drawing.Point(12, 238);
            this.tabControl1.Name = "tabControl1";
            this.tabControl1.SelectedIndex = 0;
            this.tabControl1.Size = new System.Drawing.Size(317, 374);
            this.tabControl1.TabIndex = 8;
            // 
            // tabPage1
            // 
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
            this.tabPage1.Size = new System.Drawing.Size(309, 348);
            this.tabPage1.TabIndex = 0;
            this.tabPage1.Text = "Measurement";
            this.tabPage1.UseVisualStyleBackColor = true;
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
            this.tabPage2.Location = new System.Drawing.Point(4, 22);
            this.tabPage2.Name = "tabPage2";
            this.tabPage2.Padding = new System.Windows.Forms.Padding(3);
            this.tabPage2.Size = new System.Drawing.Size(309, 318);
            this.tabPage2.TabIndex = 1;
            this.tabPage2.Text = "tabPage2";
            this.tabPage2.UseVisualStyleBackColor = true;
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(341, 624);
            this.Controls.Add(this.parallelRadioBtn);
            this.Controls.Add(this.serialRadioBtn);
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
    }
}

