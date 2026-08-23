import { KnowledgeType, EvidenceLevel } from '../../science/types/scientificTypes';

export interface EvidenceRecord {
  evidence_level: string;
  knowledge_type?: string;
}

export interface KnowledgeTypeResult {
  knowledgeTypes: KnowledgeType[];
  evidenceStatus: EvidenceLevel;
}

const LEVEL_PRIORITY: Record<EvidenceLevel, number> = {
  OBSERVED: 1,
  CANDIDATE: 2,
  SUPPORTED: 3,
  CAUSALLY_VERIFIED: 4,
  FALSIFIED: 0,
};

export function computeKnowledgeTypes(evidence: EvidenceRecord[]): KnowledgeTypeResult {
  const knowledgeTypes: KnowledgeType[] = [];
  const evidenceLevels: EvidenceLevel[] = [];
  
  for (const e of evidence) {
    const level = e.evidence_level as EvidenceLevel;
    evidenceLevels.push(level);
    
    if (level === 'CAUSALLY_VERIFIED' && e.knowledge_type === 'CAUSAL_EVIDENCE') {
      knowledgeTypes.push('CAUSAL_EVIDENCE');
    } else if (level === 'SUPPORTED' && e.knowledge_type === 'CAUSAL_EVIDENCE') {
      knowledgeTypes.push('CAUSAL_EVIDENCE');
    } else if (level === 'OBSERVED') {
      knowledgeTypes.push('OBSERVATION');
    } else if (level === 'CANDIDATE') {
      knowledgeTypes.push('INFERENCE');
    }
  }
  
  if (knowledgeTypes.length === 0) {
    knowledgeTypes.push('OBSERVATION');
  }

  let evidenceStatus: EvidenceLevel = 'OBSERVED';
  
  if (evidenceLevels.includes('FALSIFIED')) {
    evidenceStatus = 'FALSIFIED';
  } else {
    let highest = 1;
    for (const el of evidenceLevels) {
      const priority = LEVEL_PRIORITY[el];
      if (priority > highest && priority !== 0) {
        highest = priority;
        evidenceStatus = el;
      }
    }
  }

  return { knowledgeTypes, evidenceStatus };
}
