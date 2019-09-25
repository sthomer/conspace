#     def categorize_arr(self, x):
#         # Array-style categorization is much slower for some reason?
#         if len(self.unigram) == 0:
#             return uuid.uuid1().hex
#         x = x.ravel()
#         labels, centroids, radii = self.vectors()
#         candidates = self.norms(x, labels, centroids, radii)
#         return candidates[0] if candidates.size else uuid.uuid1().hex
#
#     def vectors(self):
#         # TODO: Slow (40s) These can be cached in self.stats
#         # Let's hope they iterate the same way for each comprehension?
#         labels = np.array([label for label in self.locations])
#         centroids = np.array([self.locations[label].centroid
#                               for label in labels])
#         radii = np.array([self.locations[label].radius
#                           for label in labels])
#         return labels, centroids, radii
#
#     def norms(self, x, labels, centroids, radii):
#         # TODO: VERY slow (190s) Variances can be cached in self.stats
#         distances = norm(centroids - x, axis=1)
#         candidates = labels[np.nonzero(distances <= radii)]
#         return candidates


# Either results in everything categorized together,
#   or nothing categorized together, depending on prior radius
# def categorize_union(self, x):
#     candidates = []
#     for label in self.unigram:
#         if label is None:
#             continue
#         centroid = self.stats[label].m_p
#         distance = norm(centroid - x)
#         radius = norm(np.sqrt(self.stats[label].v_p) * 3)  # i.e. 99.7%
#         if distance <= radius:
#             candidates.append(label)
#
#     category = uuid.uuid1().hex
#     if candidates:
#         # Combine stats
#         m_p = np.mean([self.stats[label].m_p for label in candidates], axis=0)
#         v_p = np.mean([self.stats[label].v_p for label in candidates], axis=0)
#         n_s = np.sum([self.unigram[label] for label in candidates])
#         m_s = np.sum([(self.unigram[label] / n_s) * self.stats[label].m_s
#                       for label in candidates], axis=0)
#         v_s = np.sum([(self.unigram[label] / n_s) * self.stats[label].v_s
#                       for label in candidates], axis=0)
#         self.stats[category] = Stats(m_s, v_s, m_p, v_p)
#         for label in candidates:
#             del self.stats[label]
#
#         # Combine unigram
#         self.unigram[category] = n_s
#         for label in candidates:
#             del self.unigram[label]
#
#         # Combine bigram: first layer
#         combined = {}
#         for label in candidates:
#             overlap = set(combined.keys()) & set(self.bigram[label].keys())
#             for c in overlap:
#                 combined[c] += self.bigram[label][c]
#                 del self.bigram[label][c]
#             combined = {**combined, **self.bigram[label]}
#             del self.bigram[label]
#         self.bigram[category] = combined
#
#         # Combine bigram: second layer
#         total = 0
#         for c in self.bigram:
#             if c is None:
#                 continue
#             for label in candidates:
#                 if label in self.bigram[c]:
#                     total += self.bigram[c][label]
#         for c in self.bigram:
#             if c is None:
#                 continue
#             for label in candidates:
#                 if label in self.bigram[c]:
#                     del self.bigram[c][label]
#                     self.bigram[c][category] = total
#
#         # Update labels in segments
#         self.current = [category if c in candidates else c for c in self.current]
#         self.segments = [[category if c in candidates else c for c in segment]
#                          for segment in self.segments]
#
#         # Remove combined candidates
#         if self.prev in candidates:
#             self.prev = category
#
#     return category
