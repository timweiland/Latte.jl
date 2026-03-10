using Test
using IntegratedNestedLaplace: owens_t
using HCubature: hcubature
using StatsFuns: erfc

@testset "owens_t" begin

    @testset "Special cases" begin
        # T(h, 0) = 0
        for h in [0.0, 0.5, 1.0, 5.0, 10.0]
            @test owens_t(h, 0.0) == 0.0
        end

        # T(0, a) = atan(a) / (2π)
        for a in [0.1, 0.5, 1.0, 2.0, 5.0]
            @test owens_t(0.0, a) ≈ atan(a) / (2π) atol = 1.0e-15
        end

        # T(h, 1) = ⅛ erfc(-h/√2) erfc(h/√2)
        invsqrt2 = 1 / sqrt(2)
        for h in [0.5, 1.0, 2.0, 3.0]
            expected = 0.125 * erfc(-h * invsqrt2) * erfc(h * invsqrt2)
            @test owens_t(h, 1.0) ≈ expected atol = 1.0e-15
        end
    end

    @testset "Symmetry" begin
        test_pairs = [(0.5, 0.3), (1.0, 0.7), (2.0, 1.5), (0.1, 3.0), (3.0, 0.1)]

        for (h, a) in test_pairs
            # T(h, -a) = -T(h, a)
            @test owens_t(h, -a) ≈ -owens_t(h, a) atol = 1.0e-15

            # T(-h, a) = T(h, a)
            @test owens_t(-h, a) ≈ owens_t(h, a) atol = 1.0e-15
        end
    end

    @testset "Reference values from StatsFuns.jl#99" begin
        # Test vectors from Andrew Gough's implementation (August 2021)
        # Reference: Patefield & Tandy (2000), Journal of Statistical Software
        hvec = [
            0.0625, 6.5, 7.0, 4.78125, 2.0, 1.0, 0.0625, 1, 1, 1, 1,
            0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25, 0.125, 0.125,
            0.125, 0.125, 0.0078125, 0.0078125, 0.0078125, 0.0078125,
            0.0078125, 0.0078125, 0.0625, 0.5, 0.9,
        ]

        avec = [
            0.25, 0.4375, 0.96875, 0.0625, 0.5, 0.9999975, 0.999999125,
            0.5, 1, 2, 3, 0.5, 1, 2, 3, 0.5, 1, 2, 3, 0.5, 1, 2, 3,
            0.5, 1, 2, 3, 10, 100, 0.999999999999999, 0.999999999999999,
            0.999999999999999,
        ]

        cvec = [
            big"0.0389119302347013668966224771378",
            big"2.00057730485083154100907167685e-11",
            big"6.399062719389853083219914429e-13",
            big"1.06329748046874638058307112826e-7",
            big"0.00862507798552150713113488319155",
            big"0.0667418089782285927715589822405",
            big"0.1246894855262192",
            big"0.04306469112078537",
            big"0.06674188216570097",
            big"0.0784681869930841",
            big"0.0792995047488726",
            big"0.06448860284750375",
            big"0.1066710629614485",
            big"0.1415806036539784",
            big"0.1510840430760184",
            big"0.07134663382271778",
            big"0.1201285306350883",
            big"0.1666128410939293",
            big"0.1847501847929859",
            big"0.07317273327500386",
            big"0.1237630544953746",
            big"0.1737438887583106",
            big"0.1951190307092811",
            big"0.07378938035365545",
            big"0.1249951430754052",
            big"0.1761984774738108",
            big"0.1987772386442824",
            big"0.2340886964802671",
            big"0.2479460829231492",
            big"0.1246895548850744",
            big"0.1066710629614484",
            big"0.0750909978020473015080760056431386286348318447478899039422181015",
        ]

        for (i, (h, a, expected)) in enumerate(zip(hvec, avec, cvec))
            result = owens_t(h, a)
            @test isapprox(result, Float64(expected); atol = 1.0e-14)
        end
    end

    @testset "Validate against quadgk" begin
        test_cases = [
            (0.5, 0.3), (1.0, 0.8), (2.0, 0.5), (3.0, 0.1),  # |a| ≤ 1
            (0.5, 1.5), (1.0, 2.0), (2.0, 3.0), (0.1, 10.0),  # |a| > 1
        ]

        for (h, a) in test_cases
            # Direct quadrature of the defining integral via hcubature
            expected, _ = hcubature(
                t -> exp(-h^2 * (1 + t[1]^2) / 2) / (1 + t[1]^2), [0.0], [a]
            )
            expected /= 2π
            @test owens_t(h, a) ≈ expected atol = 1.0e-12
        end
    end

    @testset "Large h" begin
        # T(h, a) → 0 as h → ∞
        @test owens_t(10.0, 0.5) ≈ 0.0 atol = 1.0e-20
        @test isfinite(owens_t(10.0, 0.5))
        @test isfinite(owens_t(20.0, 0.5))
        @test isfinite(owens_t(10.0, 2.0))
    end
end
