using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class OptimizationController : MonoBehaviour 
{
    public PcswiseLinear[] m_optimizablesOBJS;
    public IOptimizable[] m_optimizables;
    private ParamChanger m_changer;
    public float m_simTime=1.0f;
    private float m_currentSimTime=0.0f;
    public bool m_instantEval = true;

    private int m_currentBestCandidate=-1;
    private int m_drawBestCandidate = -1;

    private double m_lastBestScore = double.MaxValue;
    List<float> m_lastBestParams;
    public PcswiseLinear m_showcase;

    private double[] m_totalScores;
	// Use this for initialization
	void Start () 
    {
        GameObject[] objs = GameObject.FindGameObjectsWithTag("optimizable");
        m_optimizablesOBJS = new PcswiseLinear[objs.Length];
        for (int i = 0; i < objs.Length; i++)
        {
            m_optimizablesOBJS[i] = objs[i].GetComponent<PcswiseLinear>();
            m_optimizablesOBJS[i].m_initAsFunc = PcswiseLinear.INITTYPE.LIN_INC;
            m_optimizablesOBJS[i].reset(1.0f);
        }
        m_changer = new ParamChanger();
        m_optimizables = new IOptimizable[m_optimizablesOBJS.Length];
        for (int i = 0; i < m_optimizables.Length; i++)
        {
            m_optimizables[i] = (IOptimizable)m_optimizablesOBJS[i];
        }
        m_totalScores = new double[m_optimizables.Length];
        ResetScores();
	}
	
	// Update is called once per frame
	void Update () 
    {
        for (int i = 0; i < 200; i++)
        {
            EvaluateAll();
            m_currentSimTime += Time.deltaTime;
            if (m_instantEval || m_currentSimTime >= m_simTime)
            {
                FindCurrentBestCandidate();
                if (m_currentBestCandidate>=0)
                    Debug.Log("New best candidate was: " + m_currentBestCandidate);
                if (m_lastBestScore>0.01f)
                    PerturbParams();
                RestartSim();
                // Possible scene restart here <-
            }
        }
        m_showcase.SetParams(m_lastBestParams);
	}

    private void RestartSim()
    {
        m_currentSimTime = 0.0f;
        VoidBestCandidate();
        ResetScores();
    }

    private void FindCurrentBestCandidate()
    {
        VoidBestCandidate();
        double bestScore = m_lastBestScore;
        //Debug.Log("R"+bestScore);
        for (int i = 0; i < m_totalScores.Length; i++)
        {
            //Debug.Log("NR " + m_totalScores[i]);
            if (m_totalScores[i] < bestScore)
            {
                m_currentBestCandidate = i;
                bestScore = m_totalScores[i];
            }
        }
        m_lastBestScore = bestScore;
        if (m_currentBestCandidate>-1)
            m_drawBestCandidate = m_currentBestCandidate;
    }

    private void VoidBestCandidate()
    {
        m_currentBestCandidate = -1;
    }

    private void ResetScores()
    {
        for (int i = 0; i < m_totalScores.Length; i++)
        {
            m_totalScores[i] = 0.0f;
        }
    }

    private void PerturbParams()
    {
        // Get params from winner and use as basis for perturbing
        // Only update params if they were better than before, else reuse old
        if (m_currentBestCandidate > -1)
        {
            IOptimizable best=m_optimizables[m_currentBestCandidate];
            m_lastBestParams = best.GetParams();
        }
        // Perturb and assign to candidates
        for (int i = 0; i < m_optimizables.Length; i++)
        {
            IOptimizable opt = m_optimizables[i];
            List<float> newObjParams = m_changer.change(m_lastBestParams);
            // Debug.Log("new params for "+i+" sample at i=10:  _" + newObjParams[10]+"_");
            opt.SetParams(newObjParams);
        }
    }

    private void EvaluateAll()
    {
        for (int i = 0; i < m_optimizables.Length; i++)
        {
            m_totalScores[i] += GetCandidateFitness(i);
        }
    }

    private double GetCandidateFitness(int p_idx)
    {
        return m_optimizables[p_idx].EvaluateFitness();
    }

    void OnDrawGizmos()
    {
        if (m_drawBestCandidate>-1)
        {
            if (m_currentBestCandidate > -1)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawSphere(m_optimizablesOBJS[m_drawBestCandidate].transform.position - Vector3.up, 0.3f);
            }
            else
            {
                Gizmos.color = Color.blue;
                Gizmos.DrawSphere(m_optimizablesOBJS[m_drawBestCandidate].transform.position - Vector3.up, 0.3f);
            }
        }
    }
}
