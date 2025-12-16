# """
# Unit tests for NVIDIA 4:8 pairwise pruning function.

# This test suite verifies that the pruning algorithm correctly:
# 1. Prunes exactly 4 out of 8 elements per row
# 2. Prunes in pairs (2 pairs out of 4)
# 3. Selects the pairs with smallest combined metrics
# 4. Handles multiple rows independently
# 5. Works with different tensor shapes
# """

# import torch
# import unittest


# def pairwise_4_8_pruning(W_metric):
#     """
#     NVIDIA 4:8 pairwise pruning implementation.
    
#     Args:
#         W_metric: Input tensor of shape (rows, cols) where cols is divisible by 8
        
#     Returns:
#         W_mask: Boolean mask where True indicates elements to be pruned
#     """
#     W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    
#     for ii in range(0, W_metric.shape[1], 8):
#         # 1. Extract the block of 8 columns
#         tmp = W_metric[:, ii:(ii+8)].float()
        
#         # 2. Reshape to identify pairs (Batch, 4 pairs, 2 elements)
#         tmp_view = tmp.view(tmp.shape[0], 4, 2)
        
#         # 3. Calculate the metric for each pair (sum of magnitudes)
#         pair_metric = tmp_view.sum(dim=2)
        
#         # 4. Find the indices of the 2 smallest pairs to prune
#         _, pair_indices = torch.topk(pair_metric, k=2, dim=1, largest=False)
        
#         # 5. Create a mask for the pairs in this block
#         pair_mask = torch.zeros_like(pair_metric, dtype=torch.bool)
#         pair_mask.scatter_(1, pair_indices, True)
        
#         # 6. Expand the pair mask back to individual elements
#         element_mask = pair_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(tmp.shape)
        
#         # 7. Update the main mask
#         W_mask[:, ii:(ii+8)] = element_mask
    
#     return W_mask


# class TestPairwise48Pruning(unittest.TestCase):
#     """Test suite for NVIDIA 4:8 pairwise pruning."""
    
#     def test_basic_pruning_ratio(self):
#         """Test that exactly 4 out of 8 elements are pruned per row."""
#         # Create a simple 8-column tensor
#         W_metric = torch.randn(5, 8).abs()
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Check that each row has exactly 4 elements pruned
#         for row in range(W_mask.shape[0]):
#             self.assertEqual(W_mask[row].sum().item(), 4,
#                            f"Row {row} should have exactly 4 pruned elements")
    
#     def test_pairwise_pruning(self):
#         """Test that pruning happens in pairs."""
#         # Create tensor where we can control pair values
#         W_metric = torch.tensor([
#             [10.0, 10.0,  # Pair 0: sum=20 (keep)
#              1.0, 1.0,    # Pair 1: sum=2  (prune)
#              5.0, 5.0,    # Pair 2: sum=10 (keep)
#              2.0, 2.0]    # Pair 3: sum=4  (prune)
#         ])
        
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Pairs 1 and 3 should be pruned (indices 2,3 and 6,7)
#         expected_mask = torch.tensor([
#             [False, False, True, True, False, False, True, True]
#         ])
        
#         self.assertTrue(torch.equal(W_mask, expected_mask),
#                        f"Expected mask {expected_mask} but got {W_mask}")
    
#     def test_smallest_pairs_selected(self):
#         """Test that the two smallest pairs are selected for pruning."""
#         W_metric = torch.tensor([
#             [9.0, 9.0,    # Pair 0: sum=18 (keep)
#              3.0, 4.0,    # Pair 1: sum=7  (prune)
#              8.0, 7.0,    # Pair 2: sum=15 (keep)
#              1.0, 2.0]    # Pair 3: sum=3  (prune - smallest)
#         ])
        
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Should prune pairs 1 and 3
#         self.assertTrue(W_mask[0, 2] and W_mask[0, 3], "Pair 1 should be pruned")
#         self.assertTrue(W_mask[0, 6] and W_mask[0, 7], "Pair 3 should be pruned")
#         self.assertFalse(W_mask[0, 0] or W_mask[0, 1], "Pair 0 should be kept")
#         self.assertFalse(W_mask[0, 4] or W_mask[0, 5], "Pair 2 should be kept")
    
#     def test_multiple_rows_independent(self):
#         """Test that each row is pruned independently."""
#         W_metric = torch.tensor([
#             # Row 0: pairs with sums [20, 2, 10, 4] -> prune pairs 1,3
#             [10.0, 10.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0],
#             # Row 1: pairs with sums [2, 20, 4, 10] -> prune pairs 0,2
#             [1.0, 1.0, 10.0, 10.0, 2.0, 2.0, 5.0, 5.0]
#         ])
        
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Row 0: prune indices 2,3,6,7
#         self.assertFalse(W_mask[0, 0] or W_mask[0, 1])
#         self.assertTrue(W_mask[0, 2] and W_mask[0, 3])
#         self.assertFalse(W_mask[0, 4] or W_mask[0, 5])
#         self.assertTrue(W_mask[0, 6] and W_mask[0, 7])
        
#         # Row 1: prune indices 0,1,4,5
#         self.assertTrue(W_mask[1, 0] and W_mask[1, 1])
#         self.assertFalse(W_mask[1, 2] or W_mask[1, 3])
#         self.assertTrue(W_mask[1, 4] and W_mask[1, 5])
#         self.assertFalse(W_mask[1, 6] or W_mask[1, 7])
    
#     def test_multiple_blocks(self):
#         """Test pruning with multiple 8-column blocks."""
#         # Create a 16-column tensor (2 blocks of 8)
#         W_metric = torch.randn(3, 16).abs()
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Each row should have 4 pruned elements per 8-column block
#         for row in range(W_mask.shape[0]):
#             block1_pruned = W_mask[row, 0:8].sum().item()
#             block2_pruned = W_mask[row, 8:16].sum().item()
#             self.assertEqual(block1_pruned, 4, f"Row {row}, block 1 should have 4 pruned")
#             self.assertEqual(block2_pruned, 4, f"Row {row}, block 2 should have 4 pruned")
    
#     def test_deterministic_ties(self):
#         """Test behavior when pairs have equal metrics."""
#         # All pairs have the same sum - should still prune exactly 2 pairs
#         W_metric = torch.tensor([
#             [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
#         ])
        
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Should still prune exactly 4 elements (2 pairs)
#         self.assertEqual(W_mask.sum().item(), 4,
#                         "Should prune exactly 4 elements even with ties")
    
#     def test_edge_case_all_zeros(self):
#         """Test with all zero metrics."""
#         W_metric = torch.zeros(2, 8)
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Should still prune exactly 4 per row
#         for row in range(W_mask.shape[0]):
#             self.assertEqual(W_mask[row].sum().item(), 4)
    
#     def test_large_tensor(self):
#         """Test with a larger realistic tensor size."""
#         # Typical layer size
#         W_metric = torch.randn(4096, 4096).abs()
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Verify overall sparsity is 50% (4 out of 8)
#         total_elements = W_metric.numel()
#         pruned_elements = W_mask.sum().item()
#         sparsity = pruned_elements / total_elements
        
#         self.assertAlmostEqual(sparsity, 0.5, places=6,
#                               msg="Overall sparsity should be 50%")
    
#     def test_pair_consistency(self):
#         """Test that both elements in a pair are always pruned together."""
#         W_metric = torch.randn(10, 24).abs()
#         W_mask = pairwise_4_8_pruning(W_metric)
        
#         # Check every pair
#         for row in range(W_mask.shape[0]):
#             for col in range(0, W_mask.shape[1], 2):
#                 # Both elements in a pair should have the same mask value
#                 self.assertEqual(W_mask[row, col].item(), 
#                                W_mask[row, col+1].item(),
#                                f"Pair at ({row}, {col}:{col+1}) should have matching mask")
    
#     def test_output_shape(self):
#         """Test that output mask has the same shape as input."""
#         shapes = [(1, 8), (10, 16), (100, 32), (4, 64)]
#         for shape in shapes:
#             W_metric = torch.randn(shape).abs()
#             W_mask = pairwise_4_8_pruning(W_metric)
#             self.assertEqual(W_mask.shape, W_metric.shape,
#                            f"Output shape {W_mask.shape} should match input {W_metric.shape}")
    
#     def test_output_dtype(self):
#         """Test that output mask is boolean."""
#         W_metric = torch.randn(5, 8).abs()
#         W_mask = pairwise_4_8_pruning(W_metric)
#         self.assertEqual(W_mask.dtype, torch.bool,
#                         "Output mask should be boolean type")


# def run_visual_example():
#     """Run a visual example to demonstrate the pruning behavior."""
#     print("\n" + "="*70)
#     print("NVIDIA 4:8 Pairwise Pruning - Visual Example")
#     print("="*70)
    
#     # Create a controlled example
#     W_metric = torch.tensor([
#         [10.0, 10.0, 1.0, 1.0, 5.0, 5.0, 2.0, 2.0],  # Row 0
#         [3.0, 3.0, 9.0, 9.0, 2.0, 2.0, 8.0, 8.0]     # Row 1
#     ])
    
#     print("\nInput W_metric:")
#     print(W_metric)
    
#     print("\nPair structure (showing as [pair0, pair1, pair2, pair3]):")
#     for i, row in enumerate(W_metric):
#         pairs = row.view(4, 2)
#         pair_sums = pairs.sum(dim=1)
#         print(f"Row {i}: {[f'[{pairs[j,0]:.1f},{pairs[j,1]:.1f}]=>{pair_sums[j]:.1f}' for j in range(4)]}")
    
#     # Apply pruning
#     W_mask = pairwise_4_8_pruning(W_metric)
    
#     print("\nOutput W_mask (True = pruned):")
#     print(W_mask)
    
#     print("\nVisualization:")
#     for i, row_mask in enumerate(W_mask):
#         visual = ['X' if m else 'O' for m in row_mask]
#         pairs_visual = [f"[{visual[j]}{visual[j+1]}]" for j in range(0, 8, 2)]
#         print(f"Row {i}: {' '.join(pairs_visual)}  (O=keep, X=prune)")
    
#     print("\nVerification:")
#     for i in range(W_mask.shape[0]):
#         print(f"Row {i}: {W_mask[i].sum().item()}/8 elements pruned (50% sparsity)")
#     print("="*70 + "\n")


# if __name__ == '__main__':
#     # Run visual example first
#     run_visual_example()
    
#     # Run unit tests
#     print("\nRunning unit tests...\n")
#     unittest.main(verbosity=2)

"""
Visual examples of NVIDIA 4:8 pairwise pruning.
Shows before/after comparison to verify correct behavior.
"""

import torch


def pairwise_4_8_pruning(W_metric):
    """NVIDIA 4:8 pairwise pruning implementation."""
    W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
    
    for ii in range(0, W_metric.shape[1], 8):
        tmp = W_metric[:, ii:(ii+8)].float()
        tmp_view = tmp.view(tmp.shape[0], 4, 2)
        pair_metric = tmp_view.sum(dim=2)
        _, pair_indices = torch.topk(pair_metric, k=2, dim=1, largest=False)
        pair_mask = torch.zeros_like(pair_metric, dtype=torch.bool)
        pair_mask.scatter_(1, pair_indices, True)
        element_mask = pair_mask.unsqueeze(-1).expand(-1, -1, 2).reshape(tmp.shape)
        W_mask[:, ii:(ii+8)] = element_mask
    
    return W_mask


def visualize_pruning(W_metric, title="Example"):
    """Show before and after pruning comparison."""
    print(f"\n{'='*80}")
    print(f"{title}")
    print('='*80)
    
    # Apply pruning
    # print(W_metric)
    W_mask = pairwise_4_8_pruning(W_metric)
    print(W_mask)
    W_pruned = W_metric.clone()
    W_pruned[W_mask] = 0
    # print(W_pruned)
    
    for row_idx in range(W_metric.shape[0]):
        print(f"\nRow {row_idx}:")
        print("-" * 80)
        
        # Show original values with pair grouping
        print("BEFORE PRUNE:")
        row = W_metric[row_idx]
        for block_start in range(0, row.shape[0], 8):
            block = row[block_start:block_start+8]
            pairs = block.view(4, 2)
            pair_sums = pairs.sum(dim=1)
            
            print(f"  Block {block_start//8}: ", end="")
            for p_idx in range(4):
                print(f"[{pairs[p_idx,0]:6.2f}, {pairs[p_idx,1]:6.2f}] sum={pair_sums[p_idx]:7.2f}  ", end="")
            print()
        
        # Show which pairs were pruned
        print("\nAFTER PRUNE:")
        row_mask = W_mask[row_idx]
        row_pruned = W_pruned[row_idx]
        for block_start in range(0, row_pruned.shape[0], 8):
            block = row_pruned[block_start:block_start+8]
            mask_block = row_mask[block_start:block_start+8]
            pairs = block.view(4, 2)
            mask_pairs = mask_block.view(4, 2)
            
            print(f"  Block {block_start//8}: ", end="")
            for p_idx in range(4):
                if mask_pairs[p_idx, 0]:  # This pair was pruned
                    print(f"[  XXXX,   XXXX] PRUNED       ", end="")
                else:
                    print(f"[{pairs[p_idx,0]:6.2f}, {pairs[p_idx,1]:6.2f}] KEPT         ", end="")
            print()
        
        # Statistics
        total = row.shape[0]
        pruned = row_mask.sum().item()
        print(f"\n  → Pruned {pruned}/{total} elements ({pruned/total*100:.1f}% sparsity)")


# Example 1: Simple controlled values
print("\n" + "="*80)
print("NVIDIA 4:8 PAIRWISE PRUNING - VISUAL EXAMPLES")
print("="*80)

# Example 1: Clear case with distinct pair sums
W1 = torch.tensor([
    [10.0, 10.0,  1.0,  1.0,  5.0,  5.0,  2.0,  2.0],  # Pair sums: 20, 2, 10, 4 → prune pairs 1,3
])
visualize_pruning(W1, "Example 1: Clear distinct pair sums")

# Example 2: Different ordering
W2 = torch.tensor([
    [ 1.0,  2.0,  9.0,  8.0,  3.0,  4.0,  6.0,  7.0],  # Pair sums: 3, 17, 7, 13 → prune pairs 0,2
])
visualize_pruning(W2, "Example 2: Different value distribution")

# Example 3: Multiple rows
W3 = torch.tensor([
    [10.0, 10.0,  1.0,  1.0,  5.0,  5.0,  2.0,  2.0],  # Row 0: prune pairs 1,3
    [ 3.0,  3.0,  9.0,  9.0,  2.0,  2.0,  8.0,  8.0],  # Row 1: prune pairs 0,2
    [ 4.0,  5.0,  7.0,  8.0,  1.0,  1.0,  6.0,  6.0],  # Row 2: prune pairs 2,0
])
visualize_pruning(W3, "Example 3: Multiple rows (independent pruning)")

# Example 4: Multiple 8-column blocks
W4 = torch.tensor([
    [ 5.0,  5.0,  1.0,  1.0,  3.0,  3.0,  2.0,  2.0,  # Block 0: prune pairs 1,3
      8.0,  8.0,  4.0,  4.0,  9.0,  9.0,  3.0,  3.0], # Block 1: prune pairs 1,3
])
visualize_pruning(W4, "Example 4: Multiple 8-column blocks")

# Example 5: Random realistic values
torch.manual_seed(42)
W5 = torch.randn(2, 16).abs() * 10
visualize_pruning(W5, "Example 5: Random values (2 rows, 2 blocks each)")

