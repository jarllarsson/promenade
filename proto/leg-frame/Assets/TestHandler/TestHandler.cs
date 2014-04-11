using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class TestHandler : MonoBehaviour 
{
    public Controller[] m_optimizableControllers;
    private ParamChanger m_changer;
    public float m_simTime = 1.0f;
    private float m_currentSimTime = 0.0f;
    public bool m_instantEval = false;

    private int m_currentBestCandidate = -1;
    private int m_drawBestCandidate = -1;

    private double m_lastBestScore = double.MaxValue;
    List<float> m_lastBestParams;
    List<List<float>> m_currentParams;

    private double[] m_totalScores;
    // Use this for initialization
    void Start()
    {
        m_changer = new ParamChanger();
        m_currentParams = new List<List<float>>();
        m_totalScores = new double[m_optimizableControllers.Length];
        StoreParams();
        ResetScores();
    }

    // Update is called once per frame
    void Update()
    {
        m_currentSimTime += Time.deltaTime;
        if (m_instantEval || m_currentSimTime >= m_simTime)
        {            
            EvaluateAll();
            FindCurrentBestCandidate();
            if (m_currentBestCandidate >= 0)
                Debug.Log("New best candidate was: " + m_currentBestCandidate);
            if (m_lastBestScore > 0.01f)
                PerturbParams();
            RestartSim();
            // Possible scene restart here <-
        }
    }

    private void RestartSim()
    {
        m_currentSimTime = 0.0f;
        StoreParams();
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
        if (m_currentBestCandidate > -1)
            m_drawBestCandidate = m_currentBestCandidate;
    }

    private void VoidBestCandidate()
    {
        m_currentBestCandidate = -1;
    }

    private void StoreParams()
    {
        m_currentParams.Clear();
        for (int i = 0; i < m_optimizableControllers.Length; i++)
        {
            m_currentParams.Add(m_optimizableControllers[i].GetParams());
        }
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
            //IOptimizable best = m_optimizableControllers[m_currentBestCandidate];
            m_lastBestParams = m_currentParams[m_currentBestCandidate];
        }
        // Perturb and assign to candidates
        for (int i = 0; i < m_optimizableControllers.Length; i++)
        {
            IOptimizable opt = m_optimizableControllers[i];
            List<float> newObjParams = m_changer.change(m_lastBestParams); // different perturbation to each
            // Debug.Log("new params for "+i+" sample at i=10:  _" + newObjParams[10]+"_");
            opt.ConsumeParams(newObjParams); // consume it to controller
        }
    }

    private void EvaluateAll()
    {
        for (int i = 0; i < m_optimizableControllers.Length; i++)
        {
            m_totalScores[i] += EvaluateCandidateFitness(i);
        }
    }

    private double EvaluateCandidateFitness(int p_idx)
    {
        ControllerMovementRecorder record = m_optimizableControllers[p_idx].m_recordedData;
            // Test eval, data point distance to sin function
        return 1.0;
    }

    public void drawParamGraphs()
    {
        Vector2 s = new Vector2(transform.localScale.x,transform.localScale.y);
        if (m_currentParams!=null)
        {
            for (int i = 0; i < m_currentParams.Count; i++)
            {
                if (m_currentParams[i]!=null)
                {
                    Vector3 pos = m_optimizableControllers[i].transform.position + Vector3.up * 6;
                    drawLineGraph(m_currentParams[i], s, m_optimizableControllers[i].transform.position);
                }
            }
        }
        if (m_lastBestParams!=null)
            drawLineGraph(m_lastBestParams, s, transform.position);
    }

    public void drawLineGraph(List<float> p_graph, Vector2 p_scale, Vector3 p_wpos)
    {
        for (int n = 0; n < p_graph.Count-1; n++)
        {
            float t = (float)n / (float)p_graph.Count;
            float t1 = (float)(n+1) / (float)p_graph.Count;
            float val = p_graph[n];
            float val1 = p_graph[n+1];
            Gizmos.color=Color.Lerp(Color.cyan,Color.magenta,val*0.1f);
            Gizmos.DrawLine(p_wpos + Vector3.right * p_scale.x * t + Vector3.up * val * p_scale.y,
                p_wpos + Vector3.right * p_scale.x * t1 + Vector3.up * val1 * p_scale.y);
        }
    }

    void OnDrawGizmos()
    {
        if (m_drawBestCandidate > -1)
        {
            if (m_currentBestCandidate > -1)
            {
                Gizmos.color = Color.green;
                Gizmos.DrawSphere(m_optimizableControllers[m_drawBestCandidate].transform.position - Vector3.up, 0.3f);
            }
            else
            {
                Gizmos.color = Color.blue;
                Gizmos.DrawSphere(m_optimizableControllers[m_drawBestCandidate].transform.position - Vector3.up, 0.3f);
            }
        }
        drawParamGraphs();
    }
}
