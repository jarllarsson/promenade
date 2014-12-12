using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace LauncherApp
{
    // I was to lazy to make dll and wrappers and such.... ;(
    struct sharpSettingsDat
    {
        public bool m_fullscreen;

	    public string m_appMode;

	    public int m_wwidth;

	    public int m_wheight;

	    public string m_simMode;

	    public int m_measurementRuns;

	    public string m_pod;

	    public string m_execMode;

	    public int m_charcount_serial;

	    public int m_parallel_invocs;

	    public float m_charOffsetX;

	    public bool m_startPaused;

	    public int m_optmesSteps;

	    public float m_optW_fd, m_optW_fv, m_optW_fh, m_optW_fr, m_optW_fp;

    }


    class sharpSettingsReaderWriter
    {
        public void writeSettings(sharpSettingsDat p_settingsfile)
        {
            string exePathPrefix = Application.StartupPath;
            string path = exePathPrefix + "\\..\\settings.txt";
            List<string> rows = new List<string>(File.ReadAllLines(path));
            // go through each row, assume they are in order
            int optCounter = 0;
            for (int i = 0; i < rows.Count; i++)
            {
                if (rows[i] != "" && rows[i][0] != '#')
                {
                    // a writeable
                    switch (optCounter)
                    {
                        case 0:
                            rows[i] = p_settingsfile.m_fullscreen ? "1" : "0";
                            break;
                        case 1:
                            rows[i] = p_settingsfile.m_appMode;
                            break;
                        case 2:
                            rows[i] = p_settingsfile.m_wwidth.ToString();
                            break;
                        case 3:
                            rows[i] = p_settingsfile.m_wheight.ToString();
                            break;
                        case 4:
                            rows[i] = p_settingsfile.m_simMode;
                            break;
                        case 5:
                            rows[i] = p_settingsfile.m_measurementRuns.ToString();
                            break;
                        case 6:
                            rows[i] = p_settingsfile.m_pod;
                            break;
                        case 7:
                            rows[i] = p_settingsfile.m_execMode;
                            break;
                        case 8:
                            rows[i] = p_settingsfile.m_charcount_serial.ToString();
                            break;
                        case 9:
                            rows[i] = p_settingsfile.m_parallel_invocs.ToString();
                            break;
                        case 10:
                            rows[i] = p_settingsfile.m_charOffsetX.ToString();
                            break;
                        case 11:
                            rows[i] = p_settingsfile.m_startPaused ? "1" : "0";
                            break;
                        case 12:
                            rows[i] = p_settingsfile.m_optmesSteps.ToString();
                            break;
                        case 13:
                            rows[i] = p_settingsfile.m_optW_fd.ToString();
                            break;
                        case 14:
                            rows[i] = p_settingsfile.m_optW_fv.ToString();
                            break;
                        case 15:
                            rows[i] = p_settingsfile.m_optW_fh.ToString();
                            break;
                        case 16:
                            rows[i] = p_settingsfile.m_optW_fr.ToString();
                            break;
                        case 17:
                            rows[i] = p_settingsfile.m_optW_fp.ToString();
                            break;
                        default:
                            // do nothing
                            break;
                    }
                    optCounter++;
                }
            }
            // resave, using altered rows structure
            File.WriteAllLines(path, rows.ToArray());
        }


        public bool loadSettings(ref sharpSettingsDat p_settingsfile)
        {
            string exePathPrefix = Application.StartupPath;
            string path = exePathPrefix + "\\..\\settings.txt";
            List<string> rows = new List<string>(File.ReadAllLines(path));
            int optCounter = 0;
            for (int i = 0; i < rows.Count; i++)
            {
                if (rows[i] != "" && rows[i][0] != '#')
                {
                    // a writeable
                    switch (optCounter)
                    {
                        case 0:
                            p_settingsfile.m_fullscreen = rows[i] == "1" ? true : false;
                            break;
                        case 1:
                            p_settingsfile.m_appMode = rows[i];
                            break;
                        case 2:
                            p_settingsfile.m_wwidth = Convert.ToInt32(rows[i]);
                            break;
                        case 3:
                            p_settingsfile.m_wheight = Convert.ToInt32(rows[i]);
                            break;
                        case 4:
                            p_settingsfile.m_simMode = rows[i];
                            break;
                        case 5:
                            p_settingsfile.m_measurementRuns = Convert.ToInt32(rows[i]);
                            break;
                        case 6:
                            p_settingsfile.m_pod = rows[i];
                            break;
                        case 7:
                            p_settingsfile.m_execMode = rows[i];
                            break;
                        case 8:
                            p_settingsfile.m_charcount_serial = Convert.ToInt32(rows[i]);
                            break;
                        case 9:
                            p_settingsfile.m_parallel_invocs = Convert.ToInt32(rows[i]);
                            break;
                        case 10:
                            p_settingsfile.m_charOffsetX = Convert.ToSingle(rows[i]);
                            break;
                        case 11:
                            p_settingsfile.m_startPaused = rows[i] == "1" ? true : false;
                            break;
                        case 12:
                            p_settingsfile.m_optmesSteps = Convert.ToInt32(rows[i]);
                            break;
                        case 13:
                            p_settingsfile.m_optW_fd = Convert.ToSingle(rows[i]);
                            break;
                        case 14:
                            p_settingsfile.m_optW_fv = Convert.ToSingle(rows[i]);
                            break;
                        case 15:
                            p_settingsfile.m_optW_fh = Convert.ToSingle(rows[i]);
                            break;
                        case 16:
                            p_settingsfile.m_optW_fr = Convert.ToSingle(rows[i]);
                            break;
                        case 17:
                            p_settingsfile.m_optW_fp = Convert.ToSingle(rows[i]);
                            break;
                        default:
                            // do nothing
                            break;
                    }
                    optCounter++;
                }
            }
            return true;
        }

    } 	
}
