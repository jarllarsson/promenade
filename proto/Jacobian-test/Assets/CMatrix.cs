using UnityEngine;
using System.Collections;

[System.Serializable]
public class CMatrix
{
    public float[,] m;
    public int m_cols;
    public int m_rows;
	

    public CMatrix(int p_rows, int p_cols)
    {
        m_rows = p_rows; m_cols = p_cols;
        m=new float[m_rows,m_cols];  
    }

    public float this[int p_row,int p_col]
    {
        get
        {
            return m[p_row, p_col];
        }
        set
        {
            m[p_row, p_col] = value;
        }
    }

    public static CMatrix Mul(CMatrix p_ma, CMatrix p_mb)
    {
        if (p_ma.m_cols!=p_mb.m_rows)
            return null;
        CMatrix res = new CMatrix(p_ma.m_rows, p_mb.m_cols);
        int y = p_ma.m_cols;
        for (int i=0;i<res.m_rows;i++)
        for (int j=0;j<res.m_cols;j++)
        {
            float s = 0.0f;
            for (int x = 0; x < y; x++)
            {
                s += p_ma[i, x] * p_mb[x, j];
                // if (i == 1 && j == 0)
                // {
                //     Debug.Log(" a"+i + "," + x + " * b" + x + "," + j + " +");
                // }
            }
            // Debug.Log(i+","+j+"="+s);
            res[i, j] = s;
        }
        return res;
    }

    public static CMatrix Transpose(CMatrix p_m)
    {
        CMatrix res = new CMatrix(p_m.m_rows, p_m.m_cols);
        for (int i=0;i<p_m.m_rows;i++)
        for (int j=0;j<p_m.m_cols;j++)
        {
            res[i, j] = p_m[j, i];
        }
        return res;
    }

    public static CMatrix operator *(CMatrix p_ma, CMatrix p_mb)
    {
        return CMatrix.Mul(p_ma,p_mb);
    }

}
