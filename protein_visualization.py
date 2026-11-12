import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Blast import NCBIWWW, NCBIXML

fig, ax = plt.subplots(3,1, figsize=(10,11))

seq = 'MKIVLAIASLLALSAVYARPSSIKTFEEYKKAFNKSYATFEDEEAARKNFLESVKYVQSNGGAINHLSDLSLDEFKNRFLMSAEAFEHLKTQFDLNAETNACSINGNAPAEIDLRQMRTVTPIRMQGGCGSCWAFSGVAATESAYLAYRNQSLDLAEQELVDCASQHGCHGDTIPRGIEYIQHNGVVQESYYRYVAREQSCRRPNAQRFGISNYCQIYPPNVNKIREALAQTHSAIAVIIGIKDLDAFRHYDGRTIIQRDNGYQPNYHAVNIVGYSNAQGVDYWIVRNSWDTNWGDNGYGYFAANIDLMMIEEYPYVVIL'

arr = np.array(list(seq))
ax[0].imshow(arr[:, None] == arr[None, :], cmap='Greys')
ax[0].set_title('Self dot plot')

cys = [i for i, a in enumerate(seq) if a == 'C']
ax[1].vlines(cys, 0,1)
ax[1].set_xlim(0, len(seq))
ax[1].set_title('Cysteine Positions')

kd = {
    'A':1.8,'R':-4.5,'N':-3.5,'D':-3.5,'C':2.5,'Q':-3.5,'E':-3.5,'G':-0.4,'H':-3.2,'I':4.5,
    'L':3.8,'K':-3.9,'M':1.9,'F':2.8,'P':-1.6,'S':-0.8,'T':-0.7,'W':-0.9,'Y':-1.3,'V':4.2
}
vals = np.convolve([kd[a] for a in seq], np.ones(9)/9, mode='valid')
ax[2].plot(vals)
ax[2].axhline(0, color='gray')
ax[2].set_title('Hyrdrophobicity')

plt.tight_layout(h_pad=4)
# plt.show()

print(Counter(seq).most_common())
print(ProteinAnalysis(seq).molecular_weight())

# handle = NCBIWWW.qblast('blastp','swissprot', seq, hitlist_size=10)
# record = NCBIXML.read(handle)

# print(record)

##for aln in record.alignments:
  ##  hsp = aln.hsps[0]
   ## ident = 100 * hsp.identities /hsp.align_length
   ## print(f'{aln.title[:90]}\n')